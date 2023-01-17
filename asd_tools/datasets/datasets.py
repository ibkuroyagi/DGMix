from multiprocessing import Manager

import librosa
from torch.utils.data import Dataset


class WaveDataset(Dataset):
    """Outlier Wave dataset."""

    def __init__(
        self,
        df,
        allow_cache=False,
    ):
        """Initialize dataset.
        Each df["path] should be show the h5 files. (To speed up)
        """
        self.df = df
        # for cache
        self.allow_cache = allow_cache
        if allow_cache:
            # Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(df))]

    def __getitem__(self, idx):
        """Get specified idx items."""
        if self.allow_cache and (len(self.caches[idx]) != 0):
            return self.caches[idx]
        series = self.df.iloc[idx, :]
        items = {"path": series["path"]}
        wave, _ = librosa.load(path=series["path"], sr=16000)
        items["wave"] = wave - wave.mean()
        items["machine"] = series["machine"]
        items["section"] = series["section"]  # 0, 1, ..,5
        items["state"] = series["state"]  # normal or anomaly
        items["domain"] = series["domain"]  # source or target
        items["phase"] = series["phase"]  # train, valid or test
        return items

    def __len__(self):
        """Return dataset length."""
        return len(self.df)
