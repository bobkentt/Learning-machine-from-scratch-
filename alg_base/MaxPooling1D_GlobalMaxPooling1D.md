# MaxPooling1D和GlobalMaxPooling1D区别

说不明白，直接上代码，看打印：

```
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GlobalMaxPooling1D,MaxPooling1D

D = np.random.rand(10, 6, 10)

model = Sequential()
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(LSTM(10))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.summary()
```
打印：
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_4 (LSTM)                (None, 6, 16)             1728      
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 5, 16)             0         
_________________________________________________________________
lstm_5 (LSTM)                (None, 10)                1080      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11        
=================================================================
Total params: 2,819
Trainable params: 2,819
Non-trainable params: 0
_________________________________________________________________

```
model = Sequential()
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')

model.summary()
```

Layer (type)                 Output Shape              Param #   
=================================================================
lstm_7 (LSTM)                (None, 6, 16)             1728      
_________________________________________________________________
global_max_pooling1d_2 (Glob (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17        
=================================================================
Total params: 1,745
Trainable params: 1,745
Non-trainable params: 0
_________________________________________________________________
