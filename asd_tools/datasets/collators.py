import numpy as np
import torch


class WaveCollator(object):
    """Wave form data's collator."""

    def __init__(
        self,
        sf=16000,
        sec=3,
        shuffle=True,
    ):
        """Initialize customized collator for PyTorch DataLoader."""
        self.sf = sf
        self.sec = sec
        self.max_frame = int(sf * sec)
        self.shuffle = shuffle
        self.rng = np.random.default_rng()

    def __call__(self, batch):
        """Convert into batch tensors."""
        wave_batch, state_batch, section_batch, = (
            [],
            [],
            [],
        )
        for b in batch:
            start_frame = (
                self.rng.integers(max(len(b["wave"]) - self.max_frame, 1), size=1)[0]
                if self.shuffle
                else 0
            )
            wave_batch.append(
                torch.tensor(
                    b["wave"][start_frame : start_frame + self.max_frame],
                    dtype=torch.float,
                )
            )
            section_batch.append(b["section"])
            state_batch.append(b["state"])
        items = {
            "wave": torch.stack(wave_batch),
            "section": torch.tensor(
                np.array(section_batch).flatten(), dtype=torch.long
            ),
        }
        items["state"] = np.array(state_batch)
        return items


class WaveEvalCollator(object):
    """Customized collator for Pytorch DataLoader for feat form data in evaluation."""

    def __init__(
        self,
        sf=16000,
        sec=2.0,
    ):
        """Initialize customized collator for PyTorch DataLoader."""
        self.sf = sf
        self.sec = sec
        self.max_frames = int(sf * sec)

    def __call__(self, batch):
        """Convert into batch tensors."""
        waves = [b["wave"] for b in batch]
        items = {}
        wave_batch = [torch.tensor(wave, dtype=torch.float) for wave in waves]
        items["wave"] = torch.stack(wave_batch)
        items["section"] = np.array([b["section"] for b in batch])
        items["path"] = np.array([b["path"] for b in batch])
        items["state"] = np.array([b["state"] for b in batch])
        items["domain"] = np.array([b["domain"] for b in batch])
        items["phase"] = np.array([b["phase"] for b in batch])
        return items
