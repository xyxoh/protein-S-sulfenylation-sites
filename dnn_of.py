import os
import csv
import numpy as np
import time
import keras
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout,BatchNormalization
from datetime import datetime  # 用于计算时间
from sklearn import model_selection
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, roc_curve, auc
import torch
savepath = '/home/stu1/xyx/sulf/sulf_dnn'
model_name = 'dnn_all_feature_delate'

data = []
label = []
for line in open("/home/stu1/wjr/others/sss/iLearn-master/encoding/all_feature.txt","r"): #设置文件对象并读取每一行文件
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
x = x.reshape(x.shape[0],x.shape[1])

y = y.reshape(len(y),1)

x_train, x_val_test, y_train_label, y_val_test_label = model_selection.train_test_split( x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val_label, y_test_label = model_selection.train_test_split( x_val_test, y_val_test_label, test_size=0.333, random_state=42)
y_train = to_categorical(y_train_label)
y_val = to_categorical(y_val_label)
y_test = to_categorical(y_test_label)
print('训练集个数',x_train.shape)
print(x_val_test.shape)
print('验证集个数',x_val.shape[0])
print('测试集个数',x_test.shape)
#smo = SMOTE(random_state=42)
#x_train, y_train = smo.fit_resample(x_train, y_train)
#y_train = y_train.reshape(y_train.shape[0],1)
print('训练集个数',x_train.shape)
#x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
#x_val = x_val.reshape(x_val.shape[0],1,x_val.shape[1])
#x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])

def build_model(layer):
    '''
    layer: list
    instruction: the number of neurons in each layer
    '''
    model = Sequential()
    # set the first hidden layer and set the input dimension
    model.add(Dense(input_dim=layer[0],units=layer[1],activation='relu'))
    model.add(Dense(units=layer[2],activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=layer[3],activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=layer[4],activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model
#cw = {0: 1, 1: 10}
model = build_model([long,100,200,300,100])
#history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)
filepath = savepath+'/model/{}.h5'.format(model_name)
history = model.fit(x_train,
    y_train,
    batch_size=128,
    epochs=100,
    verbose=2,
    validation_data=(x_val,y_val),
    #class_weight=cw,
    #validation_split = 0.3,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, verbose=0, mode='auto')
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
scores = model.evaluate(x_test, y_test, verbose = 0)
print(model_name)
print('%s: %.2f%%'%(model.metrics_names[1], scores[1]*100))
y_val_pro = model.predict(x_val)
y_val_pro = torch.Tensor(y_val_pro)
y_val_pre = torch.argmax(y_val_pro, 1)
y_val_pre = y_val_pre.numpy()
matrix = confusion_matrix(y_val_label, y_val_pre, labels=[1, 0])    # 混淆矩阵
TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
sensitivity = recall_score(y_val_label, y_val_pre, labels=[1, 0])   # 敏感性
precision = precision_score(y_val_label,y_val_pre, labels=[1, 0])  # 精准性
accuracy = accuracy_score(y_val_label, y_val_pre)
specificity = TN / (TN + FP)    # 特异性 
mcc = (TP * TN - FP * FN) / math.sqrt((TP + FN)*(TP+FP)*(TN+FP)*(TN+FN))   # 马氏相关系数
fpr, tpr, _ = roc_curve(y_val_label.astype('int'),y_val_pre)
AUC = auc(fpr, tpr)
#results = model.evaluate(x_test,y_test)
print(model_name)
print('sensitivity',sensitivity,'precision',precision,'accuracy',accuracy,'specificity',specificity,'mcc',mcc,'AUC',AUC)
#print(results)
# plot the predicted curve and the original curve
# fill some zeros to get a (len, 51) array
