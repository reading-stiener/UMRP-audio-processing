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
import pandas as pd
from time_distributed_image_generator import TimeDistributedImageDataGenerator


# image dimensions
H=W=200
C = 3

df=pd.read_csv('LSTM/data/violin/raw/test_train.csv')
print(df)
datagen=TimeDistributedImageDataGenerator(time_steps = 5)

train_generator=datagen.flow_from_dataframe(
    dataframe=df, 
    directory='LSTM/data/violin/raw', 
    x_col='file_name', 
    y_col='class', 
    class_mode="categorical", 
    target_size=(H, W), 
    batch_size=20,
    subset='training'
)

# preparing the cnn model
model = Sequential()# after having Conv2D...
model.add(
    TimeDistributed(
        Conv2D(64, (3,3), activation='relu'), 
        input_shape=(5, H, W, C) # 5 images...
    )
)
model.add(
    TimeDistributed(
        Conv2D(64, (3,3), activation='relu')
    )
)# We need to have only one dimension per output
# to insert them to the LSTM layer - Flatten or use Pooling
model.add(
    TimeDistributed(
        Flatten()
    )
)# previous layer gives 5 outputs, Keras will make the job
# to configure LSTM inputs shape (5, ...)
model.add(
    LSTM(10, activation='relu', return_sequences=False)
)
# and then, common Dense layers... Dropout...
# up to you
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))# For example, for 3 outputs classes 
model.add(Dense(2, activation='sigmoid'))
model.compile('adam', loss='categorical_crossentropy')


#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit(
    x=train_generator,
    epochs=10
)