import glob
import os
from sound_processing import find_silences, visualize_rms, find_silences
import librosa
import numpy as np
import csv
import json
from pprint import pprint
import cv2
import subprocess
from video_processing import bbox_filepaths, draw_bbox, crop_video


def generate_dataset(file_name='bbox_annotations_URMP.csv'):
    """
    Generates the entire dataset for the project. Produces 13
    instrument folders with cropped video for that instrument, 
    individual track file, and silence annotations in JSON format.

    Parameters
    ----------

    filename : str
        csv file with bounding box annotation

    Returns
    -------
    
    None
    """
    inst_label = {

        'violin' : '_vn',
        'flute' : '_fl',
        'trumpet' : '_tpt', 
        'clarinet' : '_cl',
        'cello' : '_vc',
        'saxophone' : '_sax', 
        'tuba' : '_tba',
        'viola' : '_va',
        'trombone' : '_tbn', 
        'bassoon' :  '_bn',
        'horn' : '_hn', 
        'double_bass' : '_db', 
        'oboe' : '_ob'
    }

    URMP_base_path = 'URMP'

    if not os.path.exists('LSTM_dataset_4'): 
        os.mkdir('LSTM_dataset_4')

    base_path = 'LSTM_dataset_4'
    with open(file_name, 'r') as csvfile:
        reader =  csv.DictReader(csvfile)
        # count for instruments in a file 
        count = 1 
        folder =  ''
        for row in reader:
            #  Instrument label and path
            inst = row['label']
            inst_path = base_path + '/' + inst

            if not os.path.exists(inst_path): 
                os.mkdir(inst_path)

            # Using test image path to find URMP files
             
            img_path = row['img_path'].split('/')
            
            if not folder:
                folder = img_path[1]
            elif folder != img_path[1]: 
                folder = img_path[1]
                count = 1 
            else: 
                count += 1
            
            piece_list = folder.split('_')
            audio_file_name = 'AuSep_' + str(count) + inst_label[inst] + '_' + piece_list[0] + '_' + piece_list[1]+'.wav'
            video_file_name = 'Vid_' + folder + '.mp4'
            video_file_path = URMP_base_path + '/' + folder + '/' + video_file_name
            audio_file_path = URMP_base_path + '/' + folder + '/' + audio_file_name
            out_file_folder = base_path + '/' + inst + '/' + folder
            
            if not os.path.exists(out_file_folder):
                os.mkdir(out_file_folder)

            out_file_mix = out_file_folder + '/' + inst + '_'+ str(count) + "_mix" + '.mp4'
            out_file_inst = out_file_folder + '/' + inst + '_'+ str(count) + '.mp4'
            
            crop_video(
                file_path = video_file_path,
                audio_file_path = audio_file_path,
                annotation = row, 
                out_file_mix = out_file_mix,
                out_file_inst = out_file_inst 
            )

            silence = find_silences(
                file_name = audio_file_path,
                hop_length = 512, 
                frame_length = 2048, 
                threshold = 0.005,
                time=1
            )

            json_filename = out_file_folder + '/' + 'silences_'+inst+'_'+str(count)+'_.JSON'
            with open(json_filename, 'w') as json_file:
                json.dump(silence, json_file, indent=4)
                

if __name__ == "__main__":
    path = 'URMP'
    generate_dataset()