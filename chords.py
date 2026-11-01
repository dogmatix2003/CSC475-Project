import pandas as pd
import numpy as np
from pathlib import Path

# insert path to chord files below
chord_path = Path("The Beatles Annotations/chordlab")

# create a list of paths to all chord.lab files
lab_paths = [file.as_posix() for file in chord_path.rglob("*.lab") if file.is_file()]

# Used chord tracking for debugging
chords = set()

# parse each path in the lab_path list, simplify the chords used, print to a new .csv 
for path in lab_paths:      
    # read chord.lab file into datafram w/format onset time offset time chord, 
    # sep=r'\+' designates space seperated values instead of comma
    chord_df = pd.read_csv(path, sep=r'\s+', header = None) 
    chord_df_copy = chord_df.copy()

    #split the chords with : or / into seperate values
    split_cols = chord_df.iloc[:, 2].astype(str).str.split(r':|/')
    
    #update the dataframe copy where the chord isn't a simple major or minor to its base major/minor chord
    chord_df_copy.iloc[:,2] = np.where((split_cols.str[1].isna()) | (split_cols.str[1].astype(str).str.startswith('min')),
                                           np.where(split_cols.str[1].isna(), split_cols.str[0], split_cols.str[0] + ':min'), 
                                           split_cols.str[0])

    #make copy of the path for new chord file with simplified chords
    path_copy = path.strip(".lab")
    path = path_copy + "_simplified.csv"

    #print modified dataframe to .csv file
    chord_df_copy.to_csv(path, sep=',', header=False, index=False)
    
    #update chord set for tracking used chords for debugging
    chords.update(chord_df_copy.iloc[:,2])
    
print(sorted(chords))       
