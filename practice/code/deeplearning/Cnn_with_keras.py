
# coding: utf-8

# # CNN 卷积神经网络

# In[1]:


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


# # 加载数据

# In[2]:


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


print(X_train.shape)


# In[4]:


print(X_train[0])


# In[5]:


print(y_train.shape)


# In[6]:


y_test[1:10]


# 上面代码可以看到X_Train是一个28*28的矩阵，y是一个表示数字的列表

# In[7]:


# data pre-processing
X_train = X_train.reshape(-1,28, 28,1)/255.
X_test = X_test.reshape(-1, 28, 28,1)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# In[8]:


# Another way to build your CNN
model = Sequential()


# In[9]:


X_test.shape


# In[10]:


model.add(Convolution2D(
    input_shape=(28, 28, 1),
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same'
))


# In[11]:


model.add(Activation('relu'))


# In[12]:


model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    border_mode='same'
))


# In[13]:


model.add(Convolution2D(
    nb_filter=64,
    nb_row=5,
    nb_col=5,
    border_mode='same'
))


# In[14]:


model.add(Activation('relu'))


# In[15]:


model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    border_mode='same'
))


# In[16]:


# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))


# In[17]:


# Another way to define your optimizer
adam = Adam(lr=1e-4)


# In[18]:


model.compile(loss='categorical_crossentropy', 
              optimizer=adam,
              metrics=['accuracy']
)


# In[19]:

print('Training ------------')
model.fit(X_train, y_train, batch_size=32, nb_epoch=1)


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

