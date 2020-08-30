# author: Nikesh Ghimire

import argparse
from collections import deque
import tensorflow as tf
import cv2
import numpy as np
import json
import os 
import csv
import pandas
import shutil
from random import randrange
from sort_csv import create_p_np_annotations, count_p_np

def frame_out(path, inst_name, audio_file):
    """
    Produces dataset folder with correct labels: p and np. 

    Parameters:
    -----------

    path : str
        Base directory path.

    inst_name : str
        Name of instrument.

    audio_file : str
        Name of audio file.
    """

    try:
        vid = os.path.join(path, audio_file)
        cap = cv2.VideoCapture(vid)
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        global frame_num
        frame = 0
    
    except Exception as e:

        print(e)
        pass

    # out = cv2.VideoWriter('test_vid.mp4',-1,1,(600, 600))
    print('File opens!!')

    try:
        while(cap.isOpened()):
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            # dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)

            cv2.imshow('frame2',rgb)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',rgb)

            #Check playing or not playing from JSON
            print(path, ' ', audio_file)
            s_list = compile_interval(path, audio_file)
            
            playing = check_play(s_list, frame)

            dir_name = path[path[:-1].rfind('/') + 1:-1]

            if playing:
                print("wrote a file!!")
                cv2.imwrite('LSTM/data2/' + inst_name + '/opFlow/op_' + inst_name + '_' + str(frame_num) + '_p' + '.jpg', rgb)
                cv2.imwrite('LSTM/data2/' + inst_name + '/raw/raw_' + inst_name + '_' + str(frame_num) + '_p' + '.jpg', frame2)
            else:
                cv2.imwrite('LSTM/data2/' + inst_name + '/opFlow/op_' + inst_name + '_' + str(frame_num) + '_np' + '.jpg', rgb)
                cv2.imwrite('LSTM/data2/' + inst_name + '/raw/raw_' + inst_name + '_' + str(frame_num) + '_np' + '.jpg', frame2)
            
            
            frame_num = frame_num + 1
            frame = frame + 1
            prvs = next

        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(e)
        pass

def compile_interval(path, audio_filename):
    """
    Produces a list of intervals of silences from a silence
    JSON file. 

    Parameters: 
    -----------

    path : str
        Base directory of the audio file.
    
    audio_filename:
        Name of the audio file we are working on.
    """
    # getting rid of file extension
    audio_file = audio_filename[:-4]
    for i in os.listdir(path):
        if i.endswith('.JSON') and audio_file in i:
            break
    print(i)
    with open(os.path.join(path, i), 'r', encoding='utf-8-sig') as inFile:
        data = json.load(inFile)

    s_list = [[round(eval(k),2), round(v,2)] for k, v in data['silences'].items()] 
    
    return s_list

def check_play(silence_list, curr_frame):
    """
    Produces a playing, not playing boolean for a particular 
    frame.

    Parameters:
    -----------

    silence_list : str
        List of silence intervals.

    curr_frame : int
        Current frame count
    """
    curr_time = curr_frame/30

    for i in silence_list:
        if curr_time > i[0] and curr_time < i[1]:
            return False
        else:
            pass
    
    return True

def compile_csv(path, inst_name, skip):
    """
    Uses the CSV module to create a CSV list of file names for
    raw and opFlow folders.

    Parameters:
    -----------

    path : str
        Base directory.

    inst_name : str
        Instrument name.
    """
    inst_path = path + inst_name + '/'
    
    if not os.path.exists(inst_path): 
        os.mkdir(inst_path)

    create_p_np_annotations(inst_path, skip)
    print(count_p_np(os.path.join(inst_path, 'dataset', 'cleaned.csv')))

def resize(percent, path, inst_name):

    inst_path = path + inst_name

    n_playing = os.listdir(inst_path + '/n_playing')
    playing = os.listdir(inst_path + '/playing')
    test = os.listdir(inst_path + '/test')

    moved = 0

    pList = []
    npList = []

    for i in range(round(percent * len(n_playing))):
        
        r_num = randrange(0, len(n_playing))
        x = n_playing[r_num]
        while x in npList:
            r_num = randrange(0, len(n_playing))
            x = n_playing[r_num]
        npList.append(x)
        os.rename(inst_path + '/n_playing/' + x, path + inst_name + '_bckup/bckup/n_playing/' + x)
        moved += 1

    for i in range(round(percent * len(playing))):
        r_num = randrange(0, len(playing))
        x = playing[r_num]
        while x in pList:
            r_num = randrange(0, len(playing))
            x = playing[r_num]
        pList.append(x)
        os.rename(inst_path + '/playing/' + x, path + inst_name + '_bckup/bckup/playing/' + x)
        moved += 1

    print('Moved ' + str(moved) + ' files')

def resize_one(percent, path, inst_name, pnp):

    inst_path = path + inst_name

    n_playing = os.listdir(inst_path + '/n_playing')
    playing = os.listdir(inst_path + '/playing')
    test = os.listdir(inst_path + '/test')

    moved = 0

    pList = []
    npList = []

    if pnp == 'n':
        for i in range(round(percent * len(n_playing))):
            
            r_num = randrange(0, len(n_playing))
            x = n_playing[r_num]
            while x in npList:
                r_num = randrange(0, len(n_playing))
                x = n_playing[r_num]
            npList.append(x)
            os.rename(inst_path + '/n_playing/' + x, path + inst_name + '_bckup/bckup/n_playing/' + x)
            moved += 1
    elif pnp == 'p':
        for i in range(round(percent * len(playing))):
            r_num = randrange(0, len(playing))
            x = playing[r_num]
            while x in pList:
                r_num = randrange(0, len(playing))
                x = playing[r_num]
            pList.append(x)
            os.rename(inst_path + '/playing/' + x, path + inst_name + '_bckup/bckup/playing/' + x)
            moved += 1

    print('Moved ' + str(moved) + ' files')

def sameSize(path, inst_name):

    inst_path = path + inst_name

    n_playing = os.listdir(inst_path + '/n_playing')
    playing = os.listdir(inst_path + '/playing')
    
    moved = 0

    print(len(playing), 'playing')
    print(len(n_playing), 'n_playing')

    pList = []

    for i in range(len(playing) - len(n_playing)):
        
        r_num = randrange(0, len(playing))
        x = playing[r_num]
        while x in pList:
            r_num = randrange(0, len(playing))
            x = playing[r_num]
        pList.append(x)
        os.rename(inst_path + '/playing/' + x, path + inst_name + '_bckup/bckup/playing/' + x)
        moved += 1

    print('Resized playing w.r.t. n_playing --> Moved ' + str(moved) + ' files')

def move(inst_name, path, n_path, num_files, pnp):

    inst_path = path + inst_name

    n_playing = os.listdir(inst_path + '/n_playing')
    playing = os.listdir(inst_path + '/playing')
    
    moved = 0

    pList = []

    for i in range(num_files):
        
        r_num = randrange(0, len(playing))
        x = playing[r_num]
        while x in pList:
            r_num = randrange(0, len(playing))
            x = playing[r_num]
        pList.append(x)
        os.rename(inst_path + '/playing/' + x, n_path + '/n_playing/cello_moved_' + str(moved) + '.jpg')
        moved += 1

    print('Resized playing w.r.t. n_playing --> Moved ' + str(moved) + ' files')
    

def extract_flow(inst):
    for r_dir, sub_dir_list, file_list in os.walk(os.path.join('LSTM_dataset_4', inst)):
        for file_name in file_list:
            if 'mix' not in file_name and \
               'annotated' not in file_name and \
               'JSON' not in file_name:
                #print(r_dir, ' ',file_name)
                frame_out(r_dir, inst, file_name)

def gTruthGene(vidPath, outPath, path):

    vs = cv2.VideoCapture(vidPath)
    # mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    writer = None
    (W, H) = (None, None)
    num = 0

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction

        output = frame.copy()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (H + 300, W + 300)).astype("float32")
        # frame -= mean

        # draw the activity on the output frame
        vid_file = vidPath.split('/')[-1]
        s_list = compile_interval(path, vid_file)
        playing = check_play(s_list, num)

        print(s_list)
        print(playing)
        num+=1

        if playing:
            text = '---- playing ----'
        else:
            text = '--- n_playing ---'
        
        cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1)


        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(outPath, fourcc, 30,
                (W, H), True)

        # write the output frame to disk
        writer.write(output)

        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


def main():
    # inst_list = [
    #     'basoon',
    #     'cello',
    #     'clarinet',
    #     'double_bass',
    #     'flute',
    #     'horn',
    #     'oboe',
    #     'saxophone',
    #     'trombone',
    #     'trumpet',
    #     'tuba',
    #     'viola',
    #     'violin'
    # ]

    # for inst in inst_list:
    #     # Traverse through the directories (BFS traversal)
    #     for r_dir, sub_dir_list, file_list in os.walk(os.path.join(base_dir, inst)):
    #         # print('r_dir ', r_dir)
    #         # print('sub_dir_list ', sub_dir_list)
    #         # print('file_list ', file_list)
    #         for file_name in file_list: 
    #             if 'mix' not in file_name and 'mp4' in file_name: 
    #                 gTruthGene(os.path.join(r_dir, file_name), os.path.join(r_dir, file_name[:-4]+'annotated.mp4'), r_dir)

    #path = '/home/camel/Documents/URMP_audio_processing/LSTM_dataset_2/violin/20_Pavane_tpt_vn_vc/'
    base_dir = '/home/camel/Documents/URMP_audio_processing/LSTM_dataset_2'
     
    global frame_num
    frame_num = 0 

    # extract_flow('violin')

    # frame_num = 0 
    # extract_flow('viola')

    # frame_num = 0 
    # extract_flow('cello')
    # print(compile_interval('cnn_set/trombone/07_GString_tpt_tbn/'))
    # print(compile_interval('cnn_set/trombone/07_GString_tpt_tbn/'))
    

    # frame_out('cnn_set/trombone/07_GString_tpt_tbn/', 'trombone')

    # resize_one(0.15, 'URMP/data/', 'violin_cello_vNN', 'n')
    # sameSize('URMP/data/', 'trumpet')
    # move('cello', 'URMP/data/', 'URMP/data/violin_cello_vNN/', 4385, 'n_playing')
    
    compile_csv('LSTM/data2/', 'violin/raw', 10)


if __name__ == '__main__':
    main()