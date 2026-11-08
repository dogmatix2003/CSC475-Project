"""Settings for the machine learning part"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Annotations in the repo
ANNOTATION_DIR = REPO_ROOT / "The Beatles Annotations" / "chordlab" / "The Beatles"

# Audio is NOT in the repo
AUDIO_DIR = Path(os.environ.get("DP4_AUDIO_DIR", REPO_ROOT / "The Beatles Audio"))

# Everything the ML code creates (features, models, scores, figures) goes here.
OUTPUT_DIR = Path(os.environ.get("DP4_ML_OUTPUT", REPO_ROOT / "ml_output"))
FEATURE_DIR = OUTPUT_DIR / "features"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

# Audio / frames
FS = 22050
HOP = 2048
FRAME_RATE = FS / HOP

# Classes
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CLASS_NAMES = PITCHES + [p + "m" for p in PITCHES] + ["N"]
NO_CHORD = 24
N_CLASSES = 25

# CQT input
CQT_FMIN_NOTE = "C1"
CQT_BINS = 144 # 6 octaves
CQT_BINS_PER_OCTAVE = 24 # 2 bins per semitone

# Defaults (can be changed on the command line)
CONTEXT = 7
N_FOLDS = 5
HMM_P_STAY = 0.9
SEED = 0
