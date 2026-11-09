"""Data augmentation: move the CQT and the chord label up or down k semitones."""
import numpy as np

from ml import config


def rotate_cqt(C, k):
    """Shift a CQT (bins x frames) up by k semitones (down if k < 0), empty edge filled with 0."""
    s = k * config.CQT_BINS_PER_OCTAVE // 12# 2 bins per semitone
    if s == 0:
        return np.asarray(C)
    out = np.zeros_like(C)
    if s > 0:
        out[s:] = C[:-s]
    else:
        out[:s] = C[-s:]
    return out


def transpose_classes(classes, k):
    """Move chord classes by k semitones: C -> C#, Am -> A#m for k = 1. N stays N."""
    c = np.asarray(classes).copy()
    major, minor = c < 12, (c >= 12) & (c < 24)
    c[major] = (c[major] + k) % 12
    c[minor] = 12 + (c[minor] - 12 + k) % 12
    return c
