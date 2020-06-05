import librosa, librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy

x, sr = librosa.load('URMP/01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav')
print(sr)
print(x.shape)
print(librosa.get_duration(x, sr))
plt.figure(figsize=(12,12))

plt.subplot(4, 2, 1)
librosa.display.waveplot(x, sr=sr)
plt.title('Wave plot')


hop_length = 512
frame_length = 2048

rms = librosa.feature.rms(x, 
                        frame_length=frame_length, 
                        hop_length=hop_length, 
                        center=True)
print(rms.shape)
rms = rms[0]

frames = range(len(rms))
plt.subplot(4, 2, 2)
t = librosa.frames_to_time(frames, 
                           sr=sr, 
                           hop_length=hop_length)
plt.plot(t, rms, 'g--')
plt.title("Root Mean Square")


S, phase = librosa.magphase(librosa.stft(x))
print(S.shape)
plt.subplot(4,2,3)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=numpy.max),
                        y_axis='amplitude', 
                        x_axis='frequency')
plt.title('Log power spectogram')

plt.subplot(4,2,4)
S = librosa.magphase(librosa.stft(x, window=numpy.ones, center=False))[0]
rms = librosa.feature.rms(S=S)
print(rms.shape)
plt.title('RMS spectogram')
plt.tight_layout()
plt.show()