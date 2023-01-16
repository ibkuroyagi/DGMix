import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from asd_tools.utils import off_diagonal
from asd_tools.models import MobileNetV2Extractor
import torchaudio.transforms as T


def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class VICReg(torch.nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=1.0,
        mlp="1280-1280-1280",
        embedding_size=320,
    ):
        super().__init__()
        self.melspectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=0.0,
            f_max=8000.0,
            pad=0,
            n_mels=n_mels,
            power=power,
            normalized=True,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.backbone = MobileNetV2Extractor()
        self.section_head = nn.Sequential(
            nn.Linear(embedding_size, 128, bias=True),
            nn.Linear(128, 6, bias=True),
        )
        self.num_features = int(mlp.split("-")[-1])
        self.projector = Projector(mlp, embedding_size)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.sim_coeff = 25
        self.std_coeff = 25
        self.cov_coeff = 1

    def forward(self, wave, section=None, get_only_embed=False, getspec=False):
        output = {}
        batch_size = len(wave) // 2
        spec = self.melspectrogram(wave).unsqueeze(1)
        if getspec:
            output["spec"] = spec
        spec = spec.expand(-1, 3, -1, -1)
        embed = self.backbone(spec)[:, :, 0, 0]
        output["embedding"] = embed
        if get_only_embed:
            return output
        lam = torch.tensor(
            np.random.beta(0.5, 0.5, batch_size), dtype=torch.float32
        ).to(spec.device)[:, None, None, None]
        z = lam * spec[:batch_size] + (1 - lam) * spec[batch_size : batch_size * 2]
        embed_z = self.backbone(z)[:, :, 0, 0]
        lam = lam.squeeze(3).squeeze(2)
        if section is not None:
            repr_loss = F.mse_loss(
                embed_z,
                lam * embed[:batch_size]
                + (1 - lam) * embed[batch_size : batch_size * 2],
            )
            output["repr_loss"] = repr_loss
            section_pred = F.log_softmax(self.section_head(embed_z), dim=1)
            section = (
                lam * section[:batch_size]
                + (1 - lam) * section[batch_size : batch_size * 2]
            )
            section_loss = self.kl_loss(section_pred, section)
            loss = repr_loss + section_loss
            output["section_pred"] = self.section_head(embed)
            output["section_loss"] = section_loss
            output["loss"] = loss
            return output
        x_y = self.projector(embed)
        x, y = x_y[:batch_size], x_y[batch_size : batch_size * 2]
        z_lambda = self.projector(embed_z)
        repr_loss = F.mse_loss(z_lambda, lam * x + (1 - lam) * y)
        output["repr_loss"] = repr_loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        output["std_loss"] = std_loss
        output["cov_loss"] = cov_loss
        output["loss"] = loss
        return output
