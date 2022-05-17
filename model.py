import keras
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D, ZeroPadding2D
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization

from keras.layers import normalization
import numpy as np

from getAndEditCsv import getImagesFromDest

width = 224
height = 224
#VGG, Resnet= 244


def mainModel():
    #0 = small dataset 1 =  full
    csv,img = getImagesFromDest(width,height,0)


    trainX, testX, trainY, testY = train_test_split(img, csv , test_size=0.3, random_state=42)

    runOwnModel(trainX,testX,trainY,testY)











def createKerasResnetModel():

    base_model = ResNet50V2(weights='imagenet',
                            include_top=False,
                            input_shape=(width, height,3))

    base_model.trainable = False

    model = Sequential([
        base_model,

        Flatten(),
        Dense(6, activation="softmax"),

    ])
    return model

def runResnetKerasModel(trainX,testX,trainY,testY):
    modelKeras = createKerasResnetModel()

    modelKeras.compile(loss='sparse_categorical_crossentropy', optimizer="ADAM",
                       metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(trainX, trainY, epochs=20,validation_data=(testX,testY))



def createKerasVGG19Model():
    base_model = VGG19( weights='imagenet',
                        include_top=False,
                        input_shape=(width, height,3),

                       )
    base_model.trainable = False

    model = Sequential([
        base_model,

        Flatten(),
        Dense(6, activation="softmax"),

    ])
    return model

def runKerasVGGModel(trainX,testX,trainY,testY):
    modelKeras = createKerasVGG19Model()

    modelKeras.compile(loss='sparse_categorical_crossentropy', optimizer="ADAM",
                       metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(trainX, trainY, epochs=20,validation_data=(testX,testY))

def createKerasAlexNetModel():

    model = Sequential([
        #1. layer
        Conv2D(96, (11, 11), input_shape=(width,height,3),
               padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2,2)),

        #2. layer
        Conv2D(128, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. layer
        ZeroPadding2D((1,1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 4. layer
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),


        # 5. layer
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 6. layer
        Flatten(),
        Dense(3072),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # 7. layer
        Dense(4096),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # 8. layer
        Dense(6),
        BatchNormalization(),
        Activation('softmax'),

    ])

    return model

def runKerasAlexNetModel(trainX,testX,trainY,testY):
    modelKeras = createKerasAlexNetModel()

    modelKeras.compile(loss='sparse_categorical_crossentropy', optimizer="ADAM",
                       metrics=['sparse_categorical_accuracy'])



    modelKeras.fit(trainX, trainY, epochs=20, validation_data=(testX, testY))



def createOwnModel():


    model = Sequential([
        Conv2D(64,(8,8), input_shape=(width, height, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Dense(128,activation='relu'),

        Conv2D(128, (8, 8), input_shape=(width, height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(128, activation='relu'),

        Conv2D(128, (4, 4), input_shape=(width, height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(128, activation='relu'),

        Conv2D(64, (4, 4), input_shape=(width, height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dense(64, activation='relu'),




        Flatten(),

        Dense(6, activation="softmax"),

    ])
    return model

def runOwnModel(trainX,testX,trainY,testY):
    modelKeras = createOwnModel()

    modelKeras.summary()

    modelKeras.compile(loss='sparse_categorical_crossentropy', optimizer="ADAM",
                       metrics=['sparse_categorical_accuracy'])

    modelKeras.fit(trainX, trainY, epochs=20, validation_data=(testX, testY))


