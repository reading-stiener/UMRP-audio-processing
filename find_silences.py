import glob
import os
from sound_processing import find_silences, visualize_rms
import librosa
import numpy as np
import csv 

# create a list of file paths from the URMP folder
def g_truth_filepaths(rootdir):
    file_paths = []

    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            # only use the ground truth files for individual instruments
            if 'AuSep' in filename:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    return file_paths

# produce all annotations in a csv file
def audio_silence_annotations(file_paths):
    fields =  ['file_name', 'silence_list', 'silence_percent']
    file_name = 'silence_annotations_URMP.csv'

    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for file in file_paths:
            x, sr = librosa.load(file)
            silence = find_silences(x=x, 
                                    sr=sr, 
                                    hop_length=512, 
                                    frame_length=2048, 
                                    threshold=0.01)
            row = {
                'file_name': file,
                'silence_list': silence[0],
                'silence_percent': silence[1]
            }
            print('wrote row for ', file)
            writer.writerow(row)


path = 'URMP'
file_paths = g_truth_filepaths(path)
audio_silence_annotations(file_paths)

