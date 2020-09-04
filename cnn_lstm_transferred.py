import keras
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout
from keras.optimizers import Adam, SGD
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from time_distributed_image_generator import TimeDistributedImageDataGenerator
import os

# defining constants
H, W, C = 200, 200, 3
time_steps = 5
batch_size = 10
epochs = 50

# create a VGG16 "model", we will use
# image with shape (200, 200, 3)
vgg = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(H, W, C)
)

# do not train first layers, I want to only train
# the 4 last layers (my own choice, up to you)
for layer in vgg.layers[:-4]:
    layer.trainable = False

# create a Sequential model
model = Sequential()

# add vgg model for 5 input images (keeping the right shape
model.add(
    TimeDistributed(vgg, input_shape=(time_steps, H, W, C))
)

# now, flatten on each output to send 5 
# outputs with one dimension to LSTM
model.add(
    TimeDistributed(
        Flatten()
    )
)

# add an LSTM network
model.add(LSTM(256, activation='relu', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
adam = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    optimizer=adam, 
    loss='binary_crossentropy',
    metrics=['accuracy']    
)

print(model.summary())

df_train = pd.read_csv('LSTM/data2/violin/opFlow/dataset/train.csv')
df_test = pd.read_csv('LSTM/data2/violin/opFlow/dataset/test.csv')

train_datagen = TimeDistributedImageDataGenerator(
    time_steps = 5,
    rescale = 1./255,
    validation_split = 0.2
)
test_datagen = TimeDistributedImageDataGenerator(rescale=1./255)   
    
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train, 
    directory='LSTM/data2/violin/opFlow', 
    x_col='file_name', 
    y_col='class', 
    class_mode="binary", 
    target_size=(H, W), 
    batch_size=batch_size,
    subset='training'
)
valid_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train, 
    directory='LSTM/data2/violin/opFlow', 
    x_col='file_name', 
    y_col='class', 
    class_mode="binary", 
    target_size=(H, W), 
    batch_size=batch_size,
    subset='validation'
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test, 
    directory='LSTM/data2/violin/opFlow', 
    x_col='file_name', 
    y_col='class', 
    class_mode="binary", 
    target_size=(H, W), 
    batch_size=batch_size
)

# defining step sizes
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
step_size_test = test_generator.n//test_generator.batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=step_size_valid
)
print(history.history)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], "g--", label='train_accuracy')
plt.plot(history.history['val_accuracy'], "b--", label='validation_accuracy')
plt.title('Train and Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig("accuracy_violin_op_flow_ts5_e50_skip.png")
plt.close()

scores = model.evaluate(
    test_generator,
    steps=step_size_test
)
print(scores)

# directory for saving trained weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_lstm_weights.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Save model and weights
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)