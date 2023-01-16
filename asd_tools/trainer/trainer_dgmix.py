import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as F
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DGMixTrainer(object):
    """Customized trainer module for GDMix training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
        train=False,
        stage="pretrain",  # or "finetune"
    ):
        """Initialize trainer."""
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.train = train
        self.stage = stage
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        if stage == "finetune":
            self.epoch_valid_y_section = np.empty(0)
            self.epoch_valid_pred_section = np.empty(0)
        self.total_train_loss = defaultdict(float)
        self.total_valid_loss = defaultdict(float)
        self.best_loss = 99999
        self.steps_per_epoch = 99999

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.epochs, total=self.config["train_max_epochs"], desc="[train]"
        )
        while True:
            self._train_epoch()
            if self.epochs % self.config["log_interval_epochs"] == 0:
                self._valid_epoch()
            self._check_save_interval()
            # check whether training is finished
            self._check_train_finish()
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path, save_model_only=True):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            save_model_only (bool): Whether to save model parameters only.
        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
            "best_loss": self.best_loss,
        }
        state_dict["model"] = self.model.state_dict()
        if not save_model_only:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict["scheduler"] = self.scheduler.state_dict()
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)
        self.last_checkpoint = checkpoint_path

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.best_loss = state_dict.get("best_loss", 99999)
            logging.info(
                f"Steps:{self.steps}, Epochs:{self.epochs}, BEST loss:{self.best_loss}"
            )
            if (self.optimizer is not None) and (
                state_dict.get("optimizer", None) is not None
            ):
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if (self.scheduler is not None) and (
                state_dict.get("scheduler", None) is not None
            ):
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        section = (
            torch.nn.functional.one_hot(batch["section"], num_classes=6)
            .float()
            .to(self.device)
        )
        wave = batch["wave"].to(self.device)
        if self.stage == "pretrain":
            y_ = self.model(wave)
        elif self.stage == "finetune":
            y_ = self.model(wave, section)
        for k, v in y_.items():
            if "loss" in k:
                self.total_train_loss[f"train/{k}"] += v.item()
        y_["loss"].backward()
        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update counts
        self.steps += 1

    def _train_epoch(self):
        """Train model one epoch."""
        self.model.train()
        for steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check whether training is finished
            if self.finish_train:
                return
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({steps_per_epoch} steps per epoch)."
        )
        self._write_to_tensorboard(
            self.total_train_loss, steps_per_epoch=steps_per_epoch
        )
        self.tqdm.update(1)
        if self.stage == "prettrain":
            # for scheduler == CosineLRScheduler
            self.scheduler.step(self.epochs)
        self.epochs += 1
        self.total_train_loss = defaultdict(float)

    def _valid_step(self, batch):
        """Validate model one step."""
        section = (
            torch.nn.functional.one_hot(batch["section"], num_classes=6)
            .float()
            .to(self.device)
        )
        wave = batch["wave"].to(self.device)
        with torch.no_grad():
            if self.stage == "pretrain":
                y_ = self.model(wave)
            elif self.stage == "finetune":
                y_ = self.model(wave, section)
        for k, v in y_.items():
            if "loss" in k:
                self.total_valid_loss[f"valid/{k}"] += v.item()
        if self.stage == "finetune":
            self.epoch_valid_y_section = np.concatenate(
                [self.epoch_valid_y_section, torch.argmax(section, dim=1).cpu().numpy()]
            )
            self.epoch_valid_pred_section = np.concatenate(
                [
                    self.epoch_valid_pred_section,
                    torch.argmax(y_["section_pred"], dim=1).cpu().numpy(),
                ]
            )

    def _valid_epoch(self):
        """Validate model one epoch."""
        self.model.eval()
        for steps_per_epoch, batch in enumerate(self.data_loader["valid"], 1):
            self._valid_step(batch)
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch validation "
            f"({steps_per_epoch} steps per epoch)."
        )
        self._write_to_tensorboard(
            self.total_valid_loss, steps_per_epoch=steps_per_epoch
        )

        self.total_valid_loss["valid/loss"] /= steps_per_epoch
        if self.best_loss > self.total_valid_loss["valid/loss"]:
            self.best_loss = self.total_valid_loss["valid/loss"]
            logging.info(f"BEST Loss is updated: {self.best_loss:.5f}")
            self.save_checkpoint(
                os.path.join(self.config["outdir"], "best_loss", "best_loss.pkl"),
                save_model_only=False,
            )

        if self.stage == "finetune":
            section_micro_f1 = f1_score(
                self.epoch_valid_y_section,
                self.epoch_valid_pred_section,
                average="micro",
            )
            self._write_to_tensorboard({"valid/section_f1": section_micro_f1})
            # reset
            self.epoch_valid_y_section = np.empty(0)
            self.epoch_valid_pred_section = np.empty(0)
        self.total_valid_loss = defaultdict(float)

    def _write_embed(self):
        embed = torch.empty((0, self.config["model_params"]["embedding_size"]))
        h, w = 10, 10
        spec = torch.empty((0, 1, h, w))
        label_list = []
        with torch.no_grad():
            for b in self.data_loader["visualize"]:
                y_ = self.model(b["wave"].to(self.device), getspec=True)
                embed = torch.cat([embed, y_["embedding"].cpu()], dim=0)
                spec = torch.cat(
                    [spec, F.resize(img=y_["spec"], size=(h, w)).cpu()], dim=0
                )
                for machine, id_, phase, domain, state in zip(
                    b["machine"],
                    b["section"],
                    b["phase"],
                    b["domain"],
                    b["state"],
                ):
                    label = (
                        machine
                        + "_"
                        + str(id_)
                        + "_"
                        + phase
                        + "_"
                        + domain
                        + "_"
                        + state
                    )
                    label_list.append(label)
        self.writer.add_embedding(
            embed,
            metadata=label_list,
            label_img=spec,
            global_step=self.epochs,
        )
        logging.info(f"Successfully add embedding @ {self.epochs} epochs.")

    def _write_to_tensorboard(self, loss, steps_per_epoch=1):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value / steps_per_epoch, self.epochs)
            logging.info(
                f"(Epochs: {self.epochs}) {key} = {value / steps_per_epoch:.5f}."
            )

    def _check_save_interval(self):
        if (self.epochs % self.config["save_interval_epochs"] == 0) and (
            self.epochs != 0
        ):
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"],
                    f"checkpoint-{self.epochs}epochs",
                    f"checkpoint-{self.epochs}epochs.pkl",
                ),
                save_model_only=False,
            )
            logging.info(f"Successfully saved checkpoint @ {self.epochs} epochs.")
            self._write_embed()

    def _check_train_finish(self):
        if self.epochs >= self.config["train_max_epochs"]:
            self.finish_train = True
