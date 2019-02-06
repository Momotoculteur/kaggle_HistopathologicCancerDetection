import pandas as pd
from glob import glob
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


def loadData(df, resizeImage,imgSize):
    X = np.zeros([df.shape[0],96,96,3], dtype=np.uint8)
    Y = np.squeeze(df.as_matrix(columns=['label']))

    for i, row in tqdm(df.iterrows(),"Chargement des images en cours : ", total=df.shape[0]):
        img= Image.open(row['path'])
        if resizeImage:
            img = img.resize(size=imgSize)


    # Converti les gradients de pixels (allant de 0 à 255) vers des gradients compris entre 0 et 1. Normalization du RGB
    X = np.asarray(X)/255.
    return X,Y




trainData = pd.read_csv('train_labels.csv', sep=',')
trainPath = './train/'
testPath = './test/'
classNumber = 2
epochs = 1000
batch_size = 16
earlyStopPatience = 5

df = pd.DataFrame({'path': glob(os.path.join(trainPath,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('\\')[1].split(".")[0])
df = df.merge(trainData, on = "id")


X,Y = loadData(df,False,'')

xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=0.2, shuffle=True)

print('DIMENSION X TRAIN ' + str(xTrain.shape))
print('DIMENSION X TEST ' + str(xTest.shape))
print('DIMENSION Y TRAIN ' + str(yTrain.shape))
print('DIMENSION Y TEST ' + str(yTest.shape))

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=earlyStopPatience, verbose=0, mode='auto')
check = ModelCheckpoint('./Model.hdf5', monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='auto')

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5),input_shape=(96, 96, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5),  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(84,  activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])

trainning = model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, validation_data=(xTest, yTest),
                      callbacks=[early, check])