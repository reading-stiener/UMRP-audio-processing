import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras import Sequential
from keras.applications.vgg16 import VGG16
import pandas as pd
from time_distributed_image_generator import TimeDistributedImageDataGenerator

# create a VGG16 "model", we will use
# image with shape (200, 200, 3)
vgg = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(200, 200, 3)
)
# do not train first layers, I want to only train
# the 4 last layers (my own choice, up to you)
for layer in vgg.layers[:-4]:
    layer.trainable = False
# create a Sequential model
model = Sequential()
# add vgg model for 5 input images (keeping the right shape
model.add(
    TimeDistributed(vgg, input_shape=(5, 200, 200, 3))
)
# now, flatten on each output to send 5 
# outputs with one dimension to LSTM
model.add(
    TimeDistributed(
        Flatten()
    )
)
model.add(LSTM(256, activation='relu', return_sequences=False))
# finalize with standard Dense, Dropout...
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss='binary_crossentropy')

print(model.summary())

df=pd.read_csv('LSTM/data/violin/raw/cleaned_dataset.csv')
print(df)
datagen=TimeDistributedImageDataGenerator(time_steps = 5)

train_generator=datagen.flow_from_dataframe(
    dataframe=df, 
    directory='LSTM/data/violin/raw', 
    x_col='file_name', 
    y_col='class', 
    class_mode="binary", 
    target_size=(200, 200), 
    batch_size=20,
    subset='training'
)
model.fit(
    x=train_generator,
    epochs=10
)