# Template matching for CSC475 Chord Recognition Project

import librosa
import json
import pandas as pd
import sys
from pathlib import Path

#set up correct path to Databases folder
path = Path(sys.path[0] + "/Databases")

#create list of paths to all .json files
chord_files = [file.as_posix() for file in path.rglob("*.json") if file.is_file()]

#parse through .json files and extract relevant chord info
for file in chord_files:
    #opening each json file and accessing the dict
    current_file = json.load(open(file, "r", encoding="utf-8"))

    #grabbing the song name to properly access the dict
    songName = str(file)
    songName = songName.removeprefix((str(path) + '/').replace("\\", '/'))
    songName = songName.removesuffix('.json')
    
    #print(current_file[songName]["features_X"])
    
    
