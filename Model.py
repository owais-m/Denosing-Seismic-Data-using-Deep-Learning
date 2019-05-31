import os
import segypy
import numpy as np
import pickle
from random import randint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten, Conv3D, Conv3DTranspose, BatchNormalization
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import Adam
from PIL import Image
import tensorflow
from IPython.display import SVG 
from keras.utils.vis_utils import model_to_dot 
import random
os.chdir(r'R:\mowais\SPROJ')


# Set verbose level
segypy.verbose=1;

# Raw Data
filename3 ='Beg_IL20-100andXL1-110.sgy';
SH = segypy.getSegyHeader(filename3);
[Raw,SH,STH]=segypy.readSegy(filename3);

#Enhanced Data
filename1 ='beg_export - TensorBasedGeometric - Enhanced Coherence.sgy';
SH = segypy.getSegyHeader(filename1);
[Data,SH,STH]=segypy.readSegy(filename1);

#Min-to-Max Scaling
maxy = np.max(Raw)
mini = np.min(Raw)
RawSD = (Raw - mini)/(maxy-mini)
maxy = np.max(Data)
mini = np.min(Data)
DataSD = (Data - mini)/(maxy-mini)

#Cubify function to divide 3d data into smaller cubes
def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

#Data Reshaping
RawSD = np.array(RawSD).reshape((351,110,81))
DataSD = np.array(DataSD).reshape((351,110,81))
#Dividing data into small cubes
#195*27*22*27
#351*8910
#351/27  # 13
#110/22  # 5
#81/27   # 3
#13*5*3  #195

RawSD= cubify(RawSD, (27,22,27))
#RawSD= uncubify(RawSD, (351,110,81))
DataSD= cubify(DataSD, (27,22,27)) #194 TRAIINING EXAMPLES

RawSD = np.array(RawSD).reshape((13,5,3,27,22,27))
DataSD = np.array(DataSD).reshape((13,5,3,27,22,27))

#

#Dividing data into training and test tests
ntr =150
nte = 151
X_train = np.array(RawSD[:,:,0:2,:,:,:])
Y_train = np.array(DataSD[:,:,0:2,:,:,:])
X_test = np.array(RawSD[:,:,2,:,:,:])
Y_test = np.array(DataSD[:,:,2,:,:,:])


#Reshaping test and training sets into 4-D for training
X_train = X_train.reshape(130,27,22,27) #13*5*2
Y_train = Y_train.reshape(130,27,22,27) #13*5*2
X_test = X_test.reshape(65,27,22,27)    #13*5*1
Y_test = Y_test.reshape(65,27,22,27)
    #13*5*1
#Adding a channel to fit into our model
X_train = np.array(X_train).reshape((130,27,22,27,1))
Y_train = np.array(Y_train).reshape((130,27,22,27,1))
X_test = np.array(X_test).reshape((65,27,22,27,1))
Y_test = np.array(Y_test).reshape((65,27,22,27,1))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: = " + str(X_train.shape))
print ("Y_train shape: = " + str(X_train.shape))
print ("X_test shape: = " + str(X_test.shape))
print ("Y_test shape: = " + str(Y_test.shape))
#Model
model = Sequential()
#Filter Size
f =5
# Filters
model.add(Conv3D(filters=3, kernel_size=(f,f,f), data_format="channels_last",padding = 'valid',activation = 'relu', input_shape=(27,22,27,1)))
model.add(BatchNormalization()) 
model.add(Conv3D(filters=12, kernel_size=(f,f,f), padding = 'valid',activation = 'relu'))
model.add(BatchNormalization())
#model.add(Conv3D(filters=15, kernel_size=(f,f,f), padding = 'valid',activation = 'relu'))
#model.add(BatchNormalization())
#model.add(Conv3D(filters=18, kernel_size=(f,f,f), padding = 'valid',activation = 'relu'))
#model.add(BatchNormalization())
#model.add(Conv3DTranspose(filters=18, kernel_size=(f,f,f), padding = 'valid', activation = 'relu'))
#model.add(BatchNormalization())
#model.add(Conv3DTranspose(filters=15, kernel_size=(f,f,f), padding = 'valid', activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv3DTranspose(filters=9, kernel_size=(f,f,f), padding = 'valid', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv3DTranspose(filters=1, kernel_size=(f,f,f), padding = 'valid', activation = 'relu'))
#Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
#plot_model(model, to_file='model.png',show_shapes=True)
model.summary()

#Training
model.fit(X_train, Y_train,batch_size=1,epochs=10)

#Loss after training on trainiing data
# - loss: 0.0229 - acc: 0.1413 

#Performance on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
#test loss reduces to 0.1368 


def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)
tests = [[np.arange(4*6*16), (4,6,16), (2,3,4)],
         [np.arange(8*8*8*8), (8,8,8,8), (2,2,2,2)]]

for arr, oldshape, newshape in tests:
    arr = arr.reshape(oldshape)
    assert np.allclose(uncubify(cubify(arr, newshape), oldshape), arr)
    # cuber = Cubify(oldshape,newshape)
    # assert np.allclose(cuber.uncubify(cuber.cubify(arr)), arr)

#Output using the trained model
pred_27_22_27 = model.predict(X_test)
pred_27_22_27 = pred_27_22_27.reshape((65,27,22,27))
pred_27_22_27 = pred_27_22_27.reshape((13,5,27,22,27))
pred_27_22_27 = pred_27_22_27.reshape((351,110,27))     #13*27,5*22,1*27
pred2d_27_22_27 = pred_27_22_27.reshape((351,2970))
#pred_27_22_27 = uncubify(pred_27_22_27, (351,110,27)) #13*27,5*22,1*27


writeSEGY("R:\mowais\SPROJdatacube.sgy",pred2d_27_22_27)