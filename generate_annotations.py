import glob
import os
from sound_processing import find_silences, visualize_rms
import librosa
import numpy as np
import csv
import json
from pprint import pprint

# create a list of file paths from the URMP folder
def audio_filepaths(rootdir):
    file_paths = []

    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            # only use the ground truth files for individual instruments
            if 'AuSep' in filename:
                file_paths.append(os.path.abspath(os.path.join(folder, filename)))
            
    return file_paths

def bbox_filepaths(rootdir):

    fields = ['img_path', 'mean_annotation']
    file_img_paths = {}
    file_json_paths = {}
    row = [] 
    ref_folder = ''
    new_row = {} 
    for r_dir, sub_dir_list, file_list in os.walk(rootdir):
        if r_dir: 
            ref_folder = r_dir
        if file_list: 
            if ".jpg" in file_list[0]:
                new_row['img_path'] = ref_folder+'/'+file_list[0]
            else:
                inst_count = instrument_count(ref_folder)
                valid_count = 0  
                mean_annotation = []
                for file in file_list:
                    rel_filepath =  ref_folder + '/' + file
                    with open(rel_filepath, "r") as f:
                        js = json.load(f)
                        if len(js) == inst_count:
                            valid_count += 1
                            if not mean_annotation:
                                mean_annotation = js
                            
                            else:
                                mean_annotation = add_annotations(mean_annotation, js, inst_count)

                mean_annotation = avg_annotations(mean_annotation, valid_count, inst_count)
                if not os.path.exists('mean_annotations'):
                    os.mkdir('mean_annotations')

                new_row['mean_annotation'] = mean_annotation
                row.append(new_row)
                new_row = {}

    file_name = 'bbox_annotations_URMP.csv'
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row)

        
def add_annotations(total, new_annotation, inst_count):
    for i in range(inst_count): 
        total[i]['bottomright']['x'] += new_annotation[i]['bottomright']['x']
        total[i]['bottomright']['y'] += new_annotation[i]['bottomright']['y']
        total[i]['topleft']['x'] += new_annotation[i]['topleft']['x']
        total[i]['topleft']['y'] += new_annotation[i]['topleft']['y']
        total[i]['confidence'] += new_annotation[i]['confidence']
    return total 

def avg_annotations(total, count, inst_count):
    for i in range(inst_count): 
        total[i]['bottomright']['x'] //= count 
        total[i]['bottomright']['y'] //= count
        total[i]['topleft']['x'] //= count
        total[i]['topleft']['y'] //= count
        total[i]['confidence'] /= count
    return total 

def instrument_count(filepath):
    instruments = [ "Vn", "Fl", "Tpt", 
                    "Cl", "Vc", "Sax", 
                    "Tba", "Va", "Tbn", 
                    "Bn", "Hn", "Db", 
                    "Ob" ]

    count = 0 
    for inst in instruments:
        count =  count + filepath.count(inst)
    return count


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


# create annotations for bounding boxes based on darkflow model trained on URMP dataset

def darkflow_annotations(file_paths):
    for file in file_paths:
        continue


path = 'URMP_JSON'
#print(instrument_count('URMP_JSON/Fugue_Vn. Vn. Va. Vc/JSONS/002263.json'))
bbox_filepaths(path)
#audio_silence_annotations(file_paths)

