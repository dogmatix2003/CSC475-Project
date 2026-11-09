"""CQT features for every song, plus pitch-shifted versions for augmentation."""
import argparse

import librosa
import numpy as np

from ml import config
from ml.data import song_table, usable_songs

DEFAULT_SHIFTS = list(range(-5, 7))#semitones all 12 keys


def compute_cqt(audio_path, n_steps=0):
    """Log-magnitude CQT (144 bins x frames). n_steps != 0 pitch-shifts the audio first."""
    y, _ = librosa.load(str(audio_path), sr=config.FS)
    if n_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=config.FS, n_steps=n_steps)
    C = librosa.cqt(y, sr=config.FS, hop_length=config.HOP,
                    fmin=librosa.note_to_hz(config.CQT_FMIN_NOTE),
                    n_bins=config.CQT_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE)
    return np.log1p(np.abs(C)).astype(np.float32)


def cqt_path(song_id, n_steps=0):
    return config.FEATURE_DIR / f"cqt_shift{n_steps:+d}" / f"{song_id}.npy"


def get_cqt(song_id, audio_path, n_steps=0, mmap=False):
    """Load the saved CQT, or compute and save it the first time."""
    path = cqt_path(song_id, n_steps)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, compute_cqt(audio_path, n_steps))
    return np.load(path, mmap_mode="r" if mmap else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--shifts", type=int, nargs="*", default=[])#shifts -2 -1 1 2
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    songs = usable_songs(song_table(args.audio_dir))
    if args.limit:
        songs = songs.head(args.limit)
    for i, r in enumerate(songs.itertuples()):
        for k in [0] + [k for k in args.shifts if k != 0]:
            get_cqt(r.song_id, r.audio_path, k)
        print(f"[{i + 1}/{len(songs)}] {r.song_id}")
