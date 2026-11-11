"""Find the songs, pair annotations with audio, turn chord labels into frame classes,
and split albums into folds."""
import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff"}
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


#labels
def label_to_class(label):
    """'C' -> 0, 'Bb' -> 10, 'A:min' -> 21, 'C#m' -> 13, 'N' -> 24. Unknown labels -> 24."""
    label = str(label).strip()
    if label in ("N", "X", "", "nan"):
        return config.NO_CHORD
    m = re.match(r"^([A-G])([#b]*)(:min|m)?$", label)
    if m is None:
        return config.NO_CHORD
    pc = (LETTER_PC[m.group(1)] + m.group(2).count("#") - m.group(2).count("b")) % 12
    return pc + 12 if m.group(3) else pc


def simplified_csv_for(lab_path):
    lab_path = Path(lab_path)
    correct = lab_path.with_name(lab_path.stem + "_simplified.csv")
    if correct.exists():
        return correct
    buggy = Path(lab_path.as_posix().strip(".lab") + "_simplified.csv")
    if buggy.exists():
        return buggy
    return None


def read_segments(csv_path):
    """Simplified CSV (start,end,label; no header) -> DataFrame with start, end, label, cls."""
    df = pd.read_csv(csv_path, header=None, names=["start", "end", "label"], sep=r"[,\s]+", engine="python", keep_default_na=False)# works with commas or spaces
    df["cls"] = [label_to_class(l) for l in df["label"]]
    return df


def frame_classes(segments, n_frames):
    """Class of every frame (frame n is centred at n * HOP / FS seconds). Gaps -> N."""
    times = np.arange(n_frames) / config.FRAME_RATE
    classes = np.full(n_frames, config.NO_CHORD, dtype=np.int64)
    starts, ends, cls = segments["start"].to_numpy(), segments["end"].to_numpy(), segments["cls"].to_numpy()
    idx = np.searchsorted(starts, times, side="right") - 1
    ok = (idx >= 0) & (times < ends[np.clip(idx, 0, None)])
    classes[ok] = cls[idx[ok]]
    return classes


#songs + audio
def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def album_of(lab_path):
    """Album folder name; the two White Album CDs count as one album."""
    return re.sub(r"CD[12]", "", Path(lab_path).parent.name)


def track_title(name):
    """'CD1_-_04_-_Ob-La-Di,_Ob-La-Da' or '04 Ob-La-Di, Ob-La-Da (Remastered 2009)' -> 'obladioblada'"""
    name = name.strip().lower()
    name = re.sub(r"[\(\[].*?[\)\]]", "", name)# (Remastered 2009), [Mono]
    name = re.sub(r"\b(\d{4} )?remaster(ed)?( \d{4})?\b|\bmono\b|\bstereo\b", "", name)
    name = re.sub(r"^(cd\s*\d+)?[\s_\-.]*\d{1,2}[\s_\-.]+", "", name)# leading track numbers
    return normalize(name)


def match_audio(labs, audio_files, min_score=0.8):
    """Matching of .lab files to audio files by track title (album name helps
    break ties). Returns {lab_path: (audio_path, score)}; unmatched labs get (None, best score)."""
    pairs = []
    for i, lab in enumerate(labs):
        t, alb = track_title(lab.stem), normalize(lab.parent.name)
        for j, a in enumerate(audio_files):
            s = difflib.SequenceMatcher(None, t, track_title(a.stem)).ratio()
            if s >= min_score - 0.1:
                s += 0.05 * difflib.SequenceMatcher(None, alb, normalize(a.parent.name)).ratio()
                pairs.append((s, i, j))
    result, used_lab, used_audio = {}, set(), set()
    for s, i, j in sorted(pairs, reverse=True):
        if i not in used_lab and j not in used_audio and s >= min_score:
            result[labs[i]] = (audio_files[j], min(s, 1.0))
            used_lab.add(i)
            used_audio.add(j)
    return result


def song_table(audio_dir=None, n_folds=config.N_FOLDS, manifest=None):
    """One row per song: song_id, album, fold, lab_path, csv_path, audio_path."""
    audio_dir = Path(audio_dir or config.AUDIO_DIR)
    labs = sorted(config.ANNOTATION_DIR.rglob("*.lab"))
    audio_files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS) \
        if audio_dir.exists() else []
    given = {}
    if manifest:
        m = pd.read_csv(manifest)
        given = {Path(r.lab_path).name: r.audio_path for r in m.itertuples()}

    # balanced folds: biggest album first, always into the fold with the fewest songs
    sizes = pd.Series([album_of(p) for p in labs]).value_counts()
    fold_size, album_fold = {f: 0 for f in range(1, n_folds + 1)}, {}
    for album, n in sizes.items():
        f = min(fold_size, key=fold_size.get)
        album_fold[album] = f
        fold_size[f] += n

    matched = match_audio([l for l in labs if l.name not in given], audio_files)
    rows = []
    for lab in labs:
        if lab.name in given:
            audio, score = Path(given[lab.name]), 1.0
        else:
            audio, score = matched.get(lab, (None, 0.0))
        csv = simplified_csv_for(lab)
        rows.append({
            "song_id": normalize(lab.parent.name)[:16] + "__" + normalize(lab.stem)[:40],
            "album": album_of(lab),
            "fold": album_fold[album_of(lab)],
            "lab_path": str(lab),
            "csv_path": str(csv) if csv else "",
            "audio_path": str(audio) if audio else "",
            "match_score": round(score, 2),
        })
    return pd.DataFrame(rows)


def usable_songs(table):
    return table[(table["audio_path"] != "") & (table["csv_path"] != "")].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    t = song_table(args.audio_dir, manifest=args.manifest)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(config.RESULTS_DIR / "songs.csv", index=False)
    print(f"{len(t)} songs, {t['album'].nunique()} albums")
    print("songs per fold:", t.groupby("fold").size().to_dict())
    print("missing simplified CSV:", (t["csv_path"] == "").sum())
    print("audio found:", (t["audio_path"] != "").sum(), "of", len(t))
    if (t["audio_path"] == "").any():
        print("\nNo audio found for (rename these files or use --manifest):")
        for p in t.loc[t["audio_path"] == "", "lab_path"]:
            print("   ", Path(p).parent.name, "/", Path(p).name)
    print(f"\nSaved {config.RESULTS_DIR / 'songs.csv'}")
