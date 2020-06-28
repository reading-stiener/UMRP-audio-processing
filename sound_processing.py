import librosa, librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
import csv
from collections import OrderedDict

def visualize_rms(x, sr, hop_length, frame_length):
    """
    Visualize root mean square energies of a sound signal.
    Produces graphs of the signal alongside rms energies 
    for comparison. 

    Parameters
    ----------

    x : nparray
        Numpy array of sound signal. 
    
    sr : int
        Sampling rate.
    
    hop_length : int
    
    frame_length : int

    Returns
    -------

    None
    """
    # figure size
    plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 1)

    # display signal
    librosa.display.waveplot(x, sr=sr)
    plt.title('Wave plot')

    # calculate root mean square
    rms = librosa.feature.rms(
        x, 
        frame_length=frame_length, 
        hop_length=hop_length, 
        center=True
    )

    rms = rms[0]

    # scale from frames to time
    frames = range(len(rms))
    plt.subplot(2, 1, 2)
    t = librosa.frames_to_time(
        frames, 
        sr=sr, 
        hop_length=hop_length
    )

    #plot
    plt.plot(t, rms, 'g--')
    plt.xlabel('Time')
    plt.title("Root Mean Square")
    plt.tight_layout()
    plt.show()

def find_silences(file_name, hop_length, frame_length, threshold):
    """
    Finds the silences in a individual piece using the root mean
    square energies of the signal. A portion of a signal is defined
    as a silence if its root mean square energy is a below the 
    given threshold for more than 3 seconds.

    Parameters
    ----------

    filename : str

    hop_length : int
    
    frame_length : int
 
    Returns
    -------
    
    silences : OrderedDict
        An ordered dictionary of start of silences a keys and the
        duration of silence as values.
    """

    # calculate root mean square
    silences = OrderedDict()

    x, sr = librosa.load(file_name)

    rms = librosa.feature.rms(
        x, 
        frame_length=frame_length, 
        hop_length=hop_length, 
        center=True
    ) 

    rms = rms[0]

    # scale from frames to time
    frames = range(len(rms))

    t = librosa.frames_to_time(
        frames, 
        sr=sr, 
        hop_length=hop_length
    )
 
    start_idx = 0
    total_silence = 0
    silence_started = False
    for idx in range(rms.size):
        if rms[idx] < threshold:
            if silence_started: 
                continue
            else:
                silence_started = True
                start_idx = idx
        else:
            if t[idx] - t[start_idx] > 3 and silence_started:
                total_silence = total_silence + (t[idx] - t[start_idx])
                silences[round(t[start_idx], 3)] = round(t[idx], 3)
     
            silence_started = False
           
    return {

        'silences' : silences,
        'silence_ratio' : round(total_silence/t[-1], 3)
    }


def audio_filepaths(rootdir):
    """
    Returns a list of relevant file paths for individual tracks in the 
    URMP dataset.

    Parameters 
    ----------

    rootdir : str
        Root directory of the URMP dataset.
    
    Returns
    -------

    file_paths : list
        A list of individual track file paths.
    """

    file_paths = []

    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            # only use the ground truth files for individual instruments
            if 'AuSep' in filename:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
    return file_paths

def audio_silence_annotations(file_paths):
    """
    Creates a csv for audio silences for each individual instruments
    in a mix
    
    Parameters
    ----------

    file_paths : list
        List of relevant file paths from the URMP dataset
    
    Returns
    -------

    None
    """

    fields =  ['file_name', 'silence_list', 'silence_percent']
    file_name = 'silence_annotations_URMP.csv'

    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for file in file_paths:
            
            silence = find_silences(
                file_name = file,
                hop_length=512, 
                frame_length=2048, 
                threshold=0.01
            )
            row = {

                'file_name': file,
                'silence_list': silence['silences'],
                'silence_percent': silence['silence_ratio']
            }

            print('wrote row for ', file)
            writer.writerow(row)

if __name__ == "__main__":
   path = 'URMP'
   file_paths = audio_filepaths(path)
   audio_silence_annotations(file_paths)


    