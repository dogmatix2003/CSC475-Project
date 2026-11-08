import numpy as np
import pandas as pd
import librosa
import libfmp.c5
import libfmp.c4
import json
from pathlib import Path

# Standard parameters for an Automatic Chord Recognition (ACR) model
TARGET_SR = 22050  # Audio downsampling target
HOP_LENGTH = 2048  # Slicing step size (samples per frame) better for HMMs

# Calculate how many audio frames exist in one second
FRAME_RATE = TARGET_SR / HOP_LENGTH  # Approx 10.77 samples per frame

def convert_flat_to_sharp(note):
    flat_dict = {'Ab':'G#', 'Bb': 'A#', 'Db':'C#', 'Eb':'D#', 'Gb':'F#'}
    minor = False
    if 'm' in note:
        minor = True
        note = note[:-1]
    try:    
        if note in flat_dict.keys():
            if minor:
                return flat_dict[note] + 'm'
            return flat_dict[note] 
    except KeyError as error:
        print("f{note} Not Found.")


 
chord_ids = libfmp.c5.get_chord_labels(ext_minor='m', nonchord= True)

# Simulated path to your .wav file (replace with your actual local file path)
#audio_path = "The Beatles Audio/01_-_I_Saw_Her_Standing_There.mp3"
audio_file_path = Path("The Beatles Audio")
audio_paths = [file.as_posix() for file in audio_file_path.rglob("*.mp3") if file.is_file()]

# Insert path to simplified .csv chord files below
#lab_path = "The Beatles Annotations/chordlab/The Beatles/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There_simplified.csv"
chord_path = Path("The Beatles Annotations/chordlab")
lab_paths = [file.as_posix() for file in chord_path.rglob("*.csv") if file.is_file()]

for audio_path in audio_paths:
    track_id = Path(audio_path).stem

    # Set path to output json file with song title
    output_path = Path("Databases/" + track_id + ".json")

    # Store inside the structured dictionary object
    chord_database = {}

    for lab in lab_paths:
        #lab_id = Path(lab).stem.removesuffix('_simplified')
        if track_id in Path(lab).stem.removesuffix('_simplified'):
            chord_df = pd.read_csv(lab, sep=r'\s+', header = None)          
    
            #convert_structure_annotation() converts timestamps to framestamps based on Fs x timestamp
            #Extract the data points manually into the format libfmp expects
            #ann = [(start, end, label) for i, (start, end, label) in df.iterrows()]
            #Pass the extracted data into libfmp's conversion tool manually
            #ann = libfmp.c4.convert_structure_annotation(ann, Fs=Fs)

            # Load audio and extract Chroma feature matrix (X)
            x, sr = librosa.load(audio_path, sr=TARGET_SR)
            x_dur = x.shape[0] / sr
            chroma = librosa.feature.chroma_cqt(y=x, sr=TARGET_SR, hop_length=HOP_LENGTH, norm=None)
            total_frames = chroma.shape[1]
            X = chroma.T # Transposes chroma to get time frames as rows (total_frames, total_chroma)

            # Convert frame indices [0, 1, 2...] into their exact timestamp midpoints
            frame_times = librosa.frames_to_time(np.arange(total_frames), sr=TARGET_SR, hop_length=HOP_LENGTH)

            # Populate Ground Truth target array (Y)
            Y = np.full((total_frames,), fill_value=24, dtype=int) # Default to 24 (No Chord)
            for idx, t in enumerate(frame_times):
                # Find rows where the frame timestamp falls between start and end times
                matched_row = chord_df[(chord_df[0] <= t) & (chord_df[1] >= t)]
                if not matched_row.empty:
                    #if chord_string contains :, strip everything past : and replace with m
                    chord_string = matched_row.iat[0,2].replace(":min", "m")
                    
                    #if chord_string contains flat 'b' convert to sharp
                    if 'b' in chord_string:
                        chord_string = convert_flat_to_sharp(chord_string)
                        
                    chord_id = chord_ids.index(chord_string)
                    Y[idx] = chord_id
            
            chord_database[track_id] = {
                "paths": {
                    "audio" : audio_path,
                    ".lab" : lab
                },
                "metadata": {
                    "sampling_rate": TARGET_SR,
                    "hop_length": HOP_LENGTH,
                    "frame rate": FRAME_RATE,
                    "total_frames": total_frames
                    },
                "data": [],
                "features_X": np.round(X, 4).tolist(),      # Aligned 2D list of chroma vectors
                "labels_Y": Y.tolist()           # Aligned 1D list of integer chord targets
            }

            for segment in chord_df.itertuples(index = False):
                #if chord_string contains :, strip everything past : and replace with m
                chord_string = segment[2].replace(":min", "m")

                #if chord_string contains flat 'b' convert to sharp
                if 'b' in chord_string:
                        chord_string = convert_flat_to_sharp(chord_string)
                
                chord_id = chord_ids.index(chord_string)
                seg_dict = {
                        "start time": segment[0],
                        "end time": segment[1],
                        "duration time": segment[1] - segment[0],
                        "chord" : chord_string, 
                        "chord map": chord_id,
                        #"fold": 0,
                    }
                chord_database[track_id]["data"].append(seg_dict)
            break

    # Export the completed dictionary to a portable JSON database file
    with open(output_path, "w") as f:
        json.dump(chord_database, f, indent=2)
            