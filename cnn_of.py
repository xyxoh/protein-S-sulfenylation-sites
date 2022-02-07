import os
import csv
import numpy as np
import time
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
from datetime import datetime  # 用于计算时间
from sklearn import model_selection
from keras.utils import to_categorical
savepath = '/home/stu1/xyx/sulf/sulf_cnn'
model_name = 'CNN_DDE_delate'

data = []
label = []
for line in open("/home/stu1/wjr/others/sss/iLearn-master/encoding/DDE.txt","r"): #设置文件对象并读取每一行文件
    line=line.strip('\n')
    #print(line)
    label.append(int(line.split('\t')[0]))
    #print(line.split('\t')[0])   
    nums = line.split('\t')[1:] 
    #print(nums)  
    nums = [float(x) for x in nums]
    #print(nums)                           
    data.append(nums)#将每一行文件加入到list中


print(np.shape(data))
#print(data)
print(np.shape(label))

data= np.array(data)
y = np.array(label)
random.seed(42)
x_1 = list(data[y==1])
x_0 = list(data[y==0])
x_new_0 = random.sample(x_0,len(x_1)+400)
y =np.hstack((np.ones(len(x_1)),np.zeros(len(x_1)+400)))
print(len(x_1))
print(len(x_0))
print(len(x_new_0))
x = np.vstack((np.array(x_1),np.array(x_new_0)))
print('转换后个数',x.shape)
long = x.shape[1]
x = x.reshape(x.shape[0],x.shape[1],1)

y = y.reshape(len(y),1)
x_train, x_val_test, y_train_label, y_val_test_label = model_selection.train_test_split( x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val_label, y_test_label = model_selection.train_test_split( x_val_test, y_val_test_label, test_size=0.333, random_state=42)

print('训练集个数',x_train.shape)
print(x_val_test.shape)
print('验证集个数',x_val.shape[0])
print('测试集个数',x_test.shape)
#smo = SMOTE(random_state=42)
#x_train, y_train = smo.fit_resample(x_train, y_train)
y_train = to_categorical(y_train_label)
y_val = to_categorical(y_val_label)
y_test = to_categorical(y_test_label)
print('训练集个数',x_train.shape)

def build_model(layer):
    '''
    layer: list
    instruction: the number of neurons in each layer
    '''
    model = Sequential()
    # set the first hidden layer and set the input dimension
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(long,
        1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])

    return model

model = build_model()
model.summary()
#history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)
filepath = savepath+'/model/{}.h5'.format(model_name)
history = model.fit(x_train,
    y_train,
    batch_size=128,
    epochs=100,
    verbose=2,
    validation_data = (x_val,y_val),
    #validation_split = 0.3,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)

##history_dict = history.history
#print(history_dict.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(6,4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(savepath+'/picture/{}.png'.format(model_name))

# do the prediction
results = model.evaluate(x_test,y_test)
print(model_name)
print(results)
# plot the predicted curve and the original curve
# fill some zeros to get a (len, 51) array
