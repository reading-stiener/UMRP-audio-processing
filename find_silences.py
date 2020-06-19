import glob
import os
from sound_processing import find_silences, visualize_rms
import librosa


def g_truth_filepaths(rootdir):
    file_paths = []

    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            if 'AuSep' in filename:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    #print(file_paths)
    return file_paths




path = 'URMP'

file_paths = g_truth_filepaths(path)

print(len(file_paths))


x, sr = librosa.load(file_paths[0])

visualize_rms(x, sr, hop_length=512, frame_length=2048)

silences = find_silences(x, sr, hop_length=512, frame_length=2048, threshold=0.01)