

## Import libraries
import tensorflow as tf
import numpy as np 
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
from tensorflow.keras import layers
from keras import regularizers
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
from tensorflow.keras.models import Model

    
x

## Model functions

def zeropad(x, fils):  
  pad1 = K.zeros_like(x)
  assert (fils % pad1.shape[2]) == 0
  num_repeat = fils // pad1.shape[2]
  for i in range(num_repeat - 1):
      x = K.concatenate([x, pad1], axis=2)
  return x 
    
def basic_block(x_in, pool_size, strides, filters, kernel_size, DP):
    x = layers.BatchNormalization(axis=-1)(x_in)
    y = layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(x_in)
    y = layers.Lambda(zeropad, arguments={'fils':filters})(y) 
    x = layers.ReLU()(x) 
    x = layers.Conv1D(filters = filters, kernel_size = kernel_size ,padding='same', kernel_regularizer='L2')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DP)(x)
    x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same', kernel_regularizer='L2')(x)
    x = layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='same')(x)
    x = layers.Add()([y,x])
    return x

# Encoder
def get_encoder(filters_initial, num_cnn, residual, input_shape, name):
  
  kernel_size = 16
  DP=0.2
  DP1=0.2
  DP2=0.2
  batch_size = 128
  pool_size=2
  strides=2
  k=0
   
  filters = filters_initial*(2**k) 
  input_signal = keras.Input(shape=input_shape, name='img')
  x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same')(input_signal) 
  x = layers.BatchNormalization(axis=-1)(x)
  x = layers.ReLU()(x)
  y = layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(x)
  y = layers.Lambda(zeropad, arguments={'fils':filters})(y) 
  
  x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same')(x)
  x = layers.BatchNormalization(axis=-1)(x)
  x = layers.ReLU()(x)
  x = layers.Dropout(DP)(x)
  x = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides=strides, padding='same')(x)
  x = layers.Add()([y,x])
  
  for i in range(num_cnn):
       if i%residual == 0:
           filters = filters_initial*(2**k)
           k = k + 1 
           strides = 2
           DP = DP1
           x = basic_block(x, pool_size, strides, filters, kernel_size, DP)
       else:
           strides = 1 
           DP= DP2   
           x = basic_block(x, pool_size, strides, filters, kernel_size, DP)
          
  x = layers.ReLU()(x)
  outputs = layers.Flatten()(x)

  return tf.keras.Model(inputs=input_signal, outputs=outputs, name = name)

def simclr_loss(_,hidden,):

    temperature = 0.1
    large_num = 1e9
    hidden = tf.math.l2_normalize(hidden, -1)  
    hidden1, hidden2 = tf.split(hidden,2,0,) 
    batch_size = tf.shape(hidden1)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    
    masks = tf.one_hot(tf.range(batch_size),batch_size,) 

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * large_num

    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num

    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    loss_a = tf.compat.v1.losses.softmax_cross_entropy(labels,tf.concat([logits_ab, logits_aa], 1),)
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(labels,tf.concat([logits_ba, logits_bb], 1),)
    
    return tf.add(loss_a, loss_b)
    
# Contrastive_v2 Metrics model
def simclr_accuracy(_, hidden):

    hidden = tf.math.l2_normalize(hidden, -1)
    large_num = 1e9
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size),batch_size,)  
    
    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True)
    logits_aa = logits_aa - masks * large_num
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True)
    logits_bb = logits_bb - masks * large_num
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True)
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True)
    
    loss_a = tf.keras.metrics.categorical_accuracy(labels,tf.concat([logits_ab, logits_aa], 1),)
    loss_b = tf.keras.metrics.categorical_accuracy(labels,tf.concat([logits_ba, logits_bb], 1),)
    return tf.add(tf.reduce_mean(loss_a), tf.reduce_mean(loss_b)) / 2
    
## Contrastive_v2 Callbacks (v1 are in the principal code)
def callbacks(patience_stop=15,patience_reduce=6):
  callbacks_list = [
    keras.callbacks.EarlyStopping(
      monitor='val_sparse_categorical_accuracy',
      mode='max',
      min_delta=1e-3,
      patience=patience_stop,
      verbose=1,
      restore_best_weights=True),    
    keras.callbacks.ReduceLROnPlateau(
      monitor='val_sparse_categorical_accuracy',
      mode='max',
      factor=0.1,
      patience=patience_reduce,
      min_lr=0.00000000001)
    ]
  return callbacks_list
  

def pre_callbacks(patience_stop=15,patience_reduce=6,model_path=6):
  callbacks_list = [  
    keras.callbacks.EarlyStopping(
            monitor='val_simclr_accuracy',
            mode='max',
            patience=patience_stop,
            verbose=1,
        ),
    keras.callbacks.ReduceLROnPlateau(
            monitor='val_simclr_accuracy', 
            mode='max',
            factor=0.1,
            patience=patience_reduce,
            min_lr=0.00000000001,
            verbose=2), 
    #keras.callbacks.ModelCheckpoint(
    #       filepath=model_path, 
    #       monitor='val_simclr_accuracy',
    #       mode='max',
    #       save_best_only=True,
    #       save_weights_only=True,
    #      verbose = 2),
    ]
  return callbacks_list
  

## Define Contrastive_v2 model 
def Contrastive_model_v2(filters_initial,num_cnn,residual,input_shape,width,projection_layers,name) -> Model:

    model = get_encoder(filters_initial,num_cnn,residual,input_shape,name)
    kernel_initializer = "he_normal"
    
    if projection_layers > 0:
      model = keras.Sequential([keras.Input(shape=input_shape),model,],)
      model.add(layers.Dense(width, activation="relu", kernel_initializer=kernel_initializer, name=f"projection_{0}",))
    for i in range(projection_layers-1):
      model.add(layers.Dense(width, kernel_initializer=kernel_initializer, name=f"projection_{i+1}",))
        
    optimizer = tf.keras.optimizers.Adam(1e-2)
    
    model.compile(
        loss=[simclr_loss],
        metrics=[simclr_accuracy],
        optimizer=optimizer,
    )
    return model
    
## Calculate F1
def calculate_f1(model,x_test,y_test):
  y_pred_for_f1 = model.predict(x_test)
  y_true_for_f1 = y_test
  y_pred_for_f1_2 = np.argmax(y_pred_for_f1, axis=-1)
  CF = confusion_matrix(y_true_for_f1,y_pred_for_f1_2)
  P2 = CF.sum(axis=0)
  R2 =  CF.sum(axis=1)
  D2 = R2+P2
  F1i_2 = 2*np.diag(CF)/D2
  F1_med_2 = F1i_2[0:2]
  F1_2 = F1_med_2.sum(axis=0)/2
  print(f'[Info]: F1 = {F1_2:.8} %')
  print(CF) 
  return F1_2

def get_augmenter(signal, ruido,signal_len):
    signal_1 = signal[:,0:signal_len]
    signal_2 = signal[:,signal_len:signal_len*2]
    noise_1 = np.random.normal(0,0.03, signal_1.shape)
    noise_2 = np.random.normal(0,0.05, signal_2.shape)
    signal_1 = signal_1 + noise_1
    signal_2 = signal_2 + noise_2
    return signal_1,signal_2
