"""PyTorch datasets: CQT frames + chord labels, with optional augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ml import config
from ml.augment import rotate_cqt, transpose_classes
from ml.data import frame_classes, read_segments
from ml.features import cqt_path, get_cqt


#loading
def load_song(row, pitch_shifts=()):
    """CQT (+ pitch-shifted CQTs if they exist) and the chord class of every frame."""
    C = get_cqt(row.song_id, row.audio_path, 0)
    cqts = {0: C}
    for k in pitch_shifts:
        if k != 0 and cqt_path(row.song_id, k).exists():
            cqts[k] = np.load(cqt_path(row.song_id, k), mmap_mode="r")# read from disk, saves memory
    classes = frame_classes(read_segments(row.csv_path), C.shape[1])
    return {"song_id": row.song_id, "cqt": cqts, "classes": classes,
            "csv_path": row.csv_path, "fold": row.fold}


def compute_norm(songs):
    """Mean and std of every CQT bin over the training songs."""
    allc = np.concatenate([s["cqt"][0] for s in songs], axis=1)
    return allc.mean(axis=1, keepdims=True).astype(np.float32), \
        (allc.std(axis=1, keepdims=True) + 1e-6).astype(np.float32)


def _fit(C, n):
    """Make a pitch-shifted CQT exactly n frames long (shifting can change it by 1)."""
    C = np.asarray(C)
    return C[:, :n] if C.shape[1] >= n else np.pad(C, ((0, 0), (0, n - C.shape[1])))


#MLP / CNN: one frame + context
class FrameDataset(Dataset):
    """One example = one frame with `context` frames on each side -> (1, 144, 2*context+1), class 0-24."""

    def __init__(self, songs, mean, std, context=config.CONTEXT, augment="none", shifts=None):
        self.c, self.augment = context, augment# augment: 'none', 'rotate' or 'pitch'
        self.shifts = list(shifts) if shifts is not None else list(range(-5, 7))
        self.mean, self.std = mean, std
        self.songs = songs
        self.index = [(i, n) for i, s in enumerate(songs) for n in range(len(s["classes"]))]# (song, frame)
        self.padded = [np.pad((s["cqt"][0] - mean) / std, ((0, 0), (context, context))).astype(np.float32)
                       for s in songs]

    def __len__(self):
        return len(self.index)

    def _patch(self, i, n, k):
        if k == 0 or self.augment == "rotate":
            p = self.padded[i][:, n:n + 2 * self.c + 1]
            return rotate_cqt(p, k) if k else p
        C = self.songs[i]["cqt"][k]# pitch: cut the patch out of the shifted CQT
        lo, hi = n - self.c, n + self.c + 1
        p = np.asarray(C[:, max(lo, 0):min(hi, C.shape[1])])
        p = np.pad(p, ((0, 0), (max(0, -lo), max(0, hi - C.shape[1]))))
        return ((p - self.mean) / self.std).astype(np.float32)

    def __getitem__(self, idx):
        i, n = self.index[idx]
        k = 0
        if self.augment == "rotate":
            k = int(np.random.choice(self.shifts))# random key for this example
        elif self.augment == "pitch":
            k = int(np.random.choice(list(self.songs[i]["cqt"].keys())))# only shifts that were computed
        patch = self._patch(i, n, k)
        label = int(transpose_classes([self.songs[i]["classes"][n]], k)[0])# label moves with the CQT
        return torch.from_numpy(np.ascontiguousarray(patch))[None], label


#CRNN: chunks of frames
class ChunkDataset(Dataset):
    """One example = `length` frames in a row -> (1, 144, length), one class per frame."""

    def __init__(self, songs, mean, std, augment="none", shifts=None, length=128, chunks_per_song=20):
        self.songs, self.mean, self.std = songs, mean, std
        self.augment, self.L = augment, length
        self.shifts = list(shifts) if shifts is not None else list(range(-5, 7))
        self.index = [i for i in range(len(songs)) for _ in range(chunks_per_song)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s = self.songs[self.index[idx]]
        n = len(s["classes"])
        start = np.random.randint(0, max(1, n - self.L))# random place in the song
        k = 0
        if self.augment == "rotate":
            k = int(np.random.choice(self.shifts))
        elif self.augment == "pitch":
            k = int(np.random.choice(list(s["cqt"].keys())))
        src = s["cqt"][0] if (k == 0 or self.augment == "rotate") else _fit(s["cqt"][k], n)
        x = (np.asarray(src[:, start:start + self.L]) - self.mean) / self.std
        if self.augment == "rotate" and k:
            x = rotate_cqt(x, k)
        y = transpose_classes(s["classes"][start:start + self.L], k)
        if x.shape[1] < self.L:# short song: pad to full length
            pad = self.L - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            y = np.pad(y, (0, pad), constant_values=-100)# -100 = ignored by the loss
        return torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(y.astype(np.int64))
