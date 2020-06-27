import glob
import os
from sound_processing import find_silences, visualize_rms
import librosa
import numpy as np
import csv
import json
from pprint import pprint
import cv2
import subprocess

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

def bbox_filepaths(rootdir):
    """
    Creates a csv file of average bounding box annotations given a root directory
    with available annotations for a subset of frames for each individual instrument 
    track. 

    Parameters 
    ----------

    rootdir : str
        Root directory for URMP annotations.
    
    Returns
    --------
    None 

    """
    
    # Setup fields for csv file
    fields = ['img_path', 'label' , 'topleft_x', 'topleft_y', 'bottomright_x', 'bottomright_y', 'confidence']
    file_img_paths = {}
    file_json_paths = {}
    row = [] 
    ref_folder = ''
    image_path = ''
    
    # Traverse through the directories (BFS traversal)
    for r_dir, sub_dir_list, file_list in os.walk(rootdir):
        if r_dir:
            # Base folder for each instrument in a mix
            ref_folder = r_dir

        if file_list:
            # Example image path in the base folder 
            if ".jpg" in file_list[0]:
                image_path = ref_folder+'/'+file_list[0]
            
            else:
                # Instrument count from base folder name
                inst_count = instrument_count(ref_folder)
                valid_count = 0  
                sum_annotation = []

                # Going through each frame annotations (JSON) 
                # for a mix     
                for file in file_list:
                    rel_filepath =  ref_folder + '/' + file
            
                    with open(rel_filepath, "r") as f:
                        js = json.load(f)

                        # Checking if annotation is valid
                        if len(js) == inst_count:
                            valid_count += 1
                            if not sum_annotation:
                                sum_annotation = js
                            
                            else:
                                sum_annotation = add_annotations(sum_annotation, js, inst_count)
                
                # Average annotation
                mean_annotation = avg_annotations(sum_annotation, valid_count, inst_count)

                if not os.path.exists('mean_annotations'):
                    os.mkdir('mean_annotations')

                # Create new row list for csv
                new_rowlist = create_rows(mean_annotation, image_path)
                row.extend(new_rowlist)

    # Write to file            
    file_name = 'bbox_annotations_URMP.csv'
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row)

def create_rows(mean_annotation, img_path):
    """
    Create a list of rows for csv given averaged annotation. 
    The order of the annotations goes from left to right for 
    consistency and associating bounding box annotation with 
    audio annotation. 

    Parameters
    ----------

    mean_annotation : list
        A list of dictionaries for each instrument in the mix
        with averaged bounding box annotations.
    
    img_path : str
        File path for the test image.

    Returns
    -------
    
    row_list : list
        List of rows (annotations) for each instrument from left to 
        right.
    """
    row_list = []
    row_dict = {'img_path': img_path}

    # Go through each instrument in the averaged annotation 
    # and create a row for it 
    for annotation in mean_annotation:
        row_dict['label'] = annotation['label']
        row_dict['confidence'] = annotation['confidence']
        row_dict['topleft_x'] = annotation['topleft']['x']
        row_dict['topleft_y'] = annotation['topleft']['y']
        row_dict['bottomright_x'] = annotation['bottomright']['x']
        row_dict['bottomright_y'] = annotation['bottomright']['y']
        row_list.append(row_dict)
        row_dict = {'img_path': img_path}

    # Sort the rowlist from left to right
    row_list = sorted(row_list, key = lambda row : row['topleft_x'])
    return row_list
     
def add_annotations(total, new_annotation, inst_count):
    """
    Add up annotation values for each instrument label

    Parameters
    ----------

    total : list
       Summed up annotation so far.
    
    new_annotation : list
        List of invidual annotations for a valid JSON file.
        A JSON file if valid if there exists an annotation for each
        instrument.

    inst_count : int
        Number of instruments in a mix.

    Returns
    -------

    total : list
        Annotation after new_annotation is added to total
    """
    for i in range(inst_count): 
        total[i]['bottomright']['x'] += new_annotation[i]['bottomright']['x']
        total[i]['bottomright']['y'] += new_annotation[i]['bottomright']['y']
        total[i]['topleft']['x'] += new_annotation[i]['topleft']['x']
        total[i]['topleft']['y'] += new_annotation[i]['topleft']['y']
        total[i]['confidence'] += new_annotation[i]['confidence']
    return total 

def avg_annotations(total, count, inst_count):
    """
    Averages annotation.

    Parameters
    ----------

    total : list
        Summed up annotation so far.

    count : int
        Count of all valid annotation so far

    inst_count : int
        Number of instruments in a mix.

    Returns
    -------

    total : list
        Averaged annotation.
    """
    for i in range(inst_count): 
        total[i]['bottomright']['x'] //= count 
        total[i]['bottomright']['y'] //= count
        total[i]['topleft']['x'] //= count
        total[i]['topleft']['y'] //= count
        total[i]['confidence'] /= count
    return total 

def instrument_count(filepath):
    """
    Counts instruments in a mix. Uses the URMP 
    file path naming convention.

    Parameters
    ----------

    filepath : str
        Mix piece file path folder.

    Returns
    -------

    count : int
        Number instruments in the piece. 
    """
    instruments = [ '_vn', '_fl', '_tpt', 
                    '_cl', '_vc', '_sax', 
                    '_tba', '_va', '_tbn', 
                    '_bn', '_hn', '_db', 
                    '_ob' ]

    count = 0 
    for inst in instruments:
        count =  count + filepath.count(inst)
    return count

def draw_bbox(file_name):
    """
    Draws a bounding box over test images using averaged annotations.

    Parameters
    ----------

    filename : str
        Test file path

    Returns
    -------

    None
    """ 
    with open(file_name, 'r') as csvfile:
        reader =  csv.DictReader(csvfile)

        for row in reader:
            #print(row)
            file_name = row['img_path']
            img = cv2.imread(file_name)

            top_left = (int(row['topleft_x']), int(row['topleft_y']))
            bottom_right = (int(row['bottomright_x']), int(row['bottomright_y']))
            print(top_left)
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            cv2.imshow('image', img)
            cv2.waitKey(0)

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

def crop_video(file_path, annotation):
    """
    Crops a video given file annotation.

    Parameters
    ----------

    file_name : str
        csv file of the bounding box annotations
    
    Returns a cropped video
    """

    file_path = '/home/camel/Documents/URMP_audio_processing/URMP/12_Spring_vn_vn_vc/Vid_12_Spring_vn_vn_vc.mp4'
    top_left_x = int(annotation['topleft_x'])
    top_left_y = int(annotation['topleft_y'])
    bottom_right_x = int(annotation['bottomright_x'])
    bottom_right_y = int(annotation['bottomright_y'])
    h = bottom_right_y - top_left_y
    w = bottom_right_x - top_left_x
    ch_dir, file =  os.path.split(file_name)
    os.chdir(ch_dir)
    subprocess.call(['ffmpeg', '-i', file, '-filter:v', 'crop='+str(w)+':'+str(h)+':'+str(top_left_x)+':'+str(top_left_y), 'crop.mp4'])
      


path = 'URMP_JSON'
#print(instrument_count('URMP_JSON/Fugue_Vn. Vn. Va. Vc/JSONS/002263.json'))
bbox_filepaths(path)
#audio_silence_annotations(file_paths)
draw_bbox('bbox_annotations_URMP.csv')
#crop_video('bbox_annotations_URMP.csv')