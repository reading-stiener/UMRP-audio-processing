from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from matplotlib import pyplot
import numpy as np

H=W=200
C = 3

# setting up video input 
video = Input(shape=(None, H, W, C),name='video_input')

# preparing the cnn model
cnn = Sequential()
cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(H, W, C)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Flatten())

# putting CNN with LSTM together
cnn.trainable = True
encoded_frame = TimeDistributed(cnn)(video)
encoded_vid = LSTM(256)(encoded_frame)
outputs = Dense(50, activation='relu')(encoded_vid)


# putting layers into CNN-LSTM model
model = Model(inputs=[video],outputs=outputs)

print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

n_samples = 1
n_frames = 50

frame_sequence = np.random.randint(0.0,255.0,size=(n_samples, n_frames, H,W,C))
print(frame_sequence.shape)

y = np.random.random(size=(50,))
y = np.reshape(y,(1,50))
print(y)

model.fit(frame_sequence, y, validation_split=0.0,shuffle=False, batch_size=1)