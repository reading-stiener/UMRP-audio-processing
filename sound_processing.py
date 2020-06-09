import librosa, librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math

x, sr = librosa.load('URMP/01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav')
print(sr)
print(x.shape)
print(librosa.get_duration(x, sr))

def visualize_rms(x, sr, hop_length, frame_length):
    # figure size
    plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 1)

    # display signal
    librosa.display.waveplot(x, sr=sr)
    plt.title('Wave plot')

    # calculate root mean square
    rms = librosa.feature.rms(x, 
                            frame_length=frame_length, 
                            hop_length=hop_length, 
                            center=True)
    print(rms.shape)
    rms = rms[0]

    # scale from frames to time
    frames = range(len(rms))
    plt.subplot(2, 1, 2)
    t = librosa.frames_to_time(frames, 
                            sr=sr, 
                            hop_length=hop_length)

    #plot
    plt.plot(t, rms, 'g--')
    plt.xlabel('Time')
    plt.title("Root Mean Square")
    plt.tight_layout()
    plt.show()

def find_silences(x, sr, hop_length, frame_length, threshold):
    # calculate root mean square
    silences = []
    rms = librosa.feature.rms(x, 
                            frame_length=frame_length, 
                            hop_length=hop_length, 
                            center=True) 
    print(rms)
    rms = rms[0]
    print(type(rms))
    # scale from frames to time
    frames = range(len(rms))
    plt.subplot(2, 1, 2)
    t = librosa.frames_to_time(frames, 
                            sr=sr, 
                            hop_length=hop_length)
    print(rms.size)
    print(t.size)
    start_idx = 0
    for idx in range(rms.size):
        if rms[idx] < threshold:
            continue
        else:
            if t[idx] - t[start_idx] > 1:
                print("Found a silence")
                silences.append((t[start_idx], t[idx]))
        
            start_idx = idx
    print("Silences", silences)
    return silences

def play_video(file_name, silences): 
    cap = cv2.VideoCapture(file_name)
    folder, vid_name = os.path.split(file_name)
    try:
        # creating a folder named data
        if not os.path.exists('data/playing'):
            os.makedirs('data/playing')
        if not os.path.exists('data/not_playing'):
            os.makedirs('data/not_playing')
    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    
    frame_per_second = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print(frame_per_second)
    current_frame = 0
    
    idx = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret:
            # if video is still left continue creating images
            print(silences[idx][0]*frame_per_second)
            print(current_frame)
            if current_frame < (silences[idx][0]*frame_per_second):
                if current_frame % (2*frame_per_second) == 0: 
                    name = 'data/playing/' + vid_name + '_' + str(current_frame) + '.jpg'
                    print('Creating...' + name)
                    cv2.imwrite(name, frame)
            elif current_frame > (silences[idx][0]*frame_per_second) \
                and current_frame < (silences[idx][1]*frame_per_second):
                if (current_frame % math.floor(frame_per_second/2)) == 0:
                    name = 'data/not_playing/' + vid_name + '_' + str(current_frame) + '.jpg'
                    print('Creating...' + name)
                    cv2.imwrite(name, frame)
            else:
                if idx < len(silences)-1:
                    idx += 1 
            current_frame += 1 
        
        else:
            break 
    print(current_frame)
    cap.release()
    cv2.destroyAllWindows()

silences = find_silences(x, sr, hop_length=512, frame_length=2048, threshold=0.02)
#pip visualize_rms(x, sr, hop_length=512, frame_length=2048)
file_name = 'URMP/01_Jupiter_vn_vc/Vid_01_Jupiter_vn_vc.mp4'
play_video(file_name, silences)