import numpy as np 
import os,cv2

import sklearn.preprocessing

from keras import backend as kb
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU,ELU

from keras.optimizers import SGD,RMSprop,adam
from sklearn import utils
from sklearn import model_selection


def imag_vec_conversion(img,size=(256,256)):
	return cv2.resize(img,size).flatten()


def count_imges(dir):

	return len([name for name in os.listdir('.') if os.path.isfile(name)])



path = os.getcwd() + '\images'
img_list = os.listdir(path)

rows = 256
cols = 256
chnls = 1
n_epochs = 25

n_classes = 2

li_img_data = []

for img_type in img_list:
	img_list = os.listdir(path + '/' + img_type)
	print('{} images loaded'.format(img_type))
	for each_img in img_list:
		img = cv2.imread(path + '/'+ img_type + '/'+ each_img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(rows,cols))
		li_img_data.append(img)

img_arr = np.array(li_img_data).astype('float32')
img_arr /=255

print(img_arr.shape)

kb.set_image_dim_ordering("tf")

img_arr = np.expand_dims(img_arr,axis=4)

n_classes = 2

n_samples = img_arr.shape[0]
labels = np.ones((n_samples),dtype='int64')

path_pos = path+'\pos'
li = os.listdir(path_pos)
print(len(li),' number of positive images')

labels[0:len(li)+1] = 0
labels[len(li)+1:] = 1

Y = np_utils.to_categorical(labels,n_classes)

x,y = utils.shuffle(img_arr,Y)

X_train,X_test,y_train,y_test = model_selection.train_test_split(x,y,test_size= 0.2,random_state=2)

input_shape = img_arr[0].shape

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),padding='same',input_shape=input_shape,activation='linear'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer = 'Adam',metrics=['accuracy'])

clf = model.fit(X_train,y_train,batch_size=16,epochs=n_epochs,verbose=1,validation_data=(X_test,y_test))

score = model.evaluate(X_test,y_test,verbose=0)
print(score[1])

model.save('face_recog.hdf5')

