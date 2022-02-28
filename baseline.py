# -*- coding: utf-8 -*-

# Own libraries
import import_data
import import_functions

import numpy as np 
import numpy.matlib
import os 
import matplotlib.pyplot as plt 

# Training libraries
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
from tensorflow.keras import layers
from keras import regularizers
from keras.layers import LayerNormalization
from sklearn.model_selection import StratifiedKFold

#  Data libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.data import AUTOTUNE
import h5py
import sys 

stdoutOrigin=sys.stdout 
sys.stdout = open("base_icentia_10K_m16.txt", "w") #File name

print('Baseline con shuffle, características por defecto, filtro inicial 16') #delete

GPU1 = "2"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1

# Hyperparameters
signal_len = 2048
labeled_samples = 10000 
num_classes = 3
batch_size = 256 

# Baseline model parameters
baseline_epochs = 100 
dense_width = 128
width =  128
filters_initial = 16
num_cnn = 13
residual = 4
kernel_size = 16
patience_stop = 15
patience_reduce = 4


print('\n[Info]: Import data')

x_train_f, y_train_f = import_data.load_data_train('simple', signal_len, labeled_samples)
print('\n[Info]: Simple train data correctly imported')

x_test, y_test = import_data.load_data_test(signal_len)
print('\n[Info]: Test data correctly imported')

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test,y_test))
          .batch(batch_size, drop_remainder = True)
          .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

print('\n[Info]: Size for x_train_f = '+str(x_train_f.shape))

## K-folds settings

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)

## Callbacks

callbacks_list = [
keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    mode='auto',
    min_delta=1e-3,
    patience=15,
    verbose=2,
    restore_best_weights=True),    
keras.callbacks.ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy',
    mode='auto',
    factor=0.1,
    patience=4,
    verbose = 2,
    min_lr=0.00000000001),
]

inputs = x_train_f
targets = y_train_f

fold_no = 1
Acc_Val_per_fold = []
F1_per_fold = []
Acc_total = []
F1_total = []
Loss_total = []

for train, test in skf.split(inputs, targets):

  print('\n[Info]: Traning for the fold ', fold_no)
  
  
  ##Baseline model
  
  print('\n[Info]: Starting training: ')
  
  x_train_fold  = inputs[train] 
  y_train_fold = targets[train]
  print('Tamaño de x_train_fold = '+str(x_train_fold.shape))
  x_val_fold = inputs[test] 
  y_val_fold = targets[test]
  print('Tamaño de validacion = '+str(x_val_fold.shape))

  classes = tf.math.bincount(y_train_fold)
  classes = np.array(classes)
  print('[Info]: x_train_fold:')
  print('Number of train samples: ',y_train_fold.shape[0])
  print('Distribution of classes: ',classes)
  
  classes = tf.math.bincount(y_val_fold)
  classes = np.array(classes)
  print('[Info]: Validation:')
  print('Number of train samples: ',y_val_fold.shape[0])
  print('Distribution of classes: ',classes)
  
  labeled_train_dataset = (tf.data.Dataset.from_tensor_slices((x_train_fold,y_train_fold))
                      .shuffle(buffer_size=10*batch_size, seed= 0)
                      .batch(batch_size, drop_remainder = True))
                      
  val_dataset = (tf.data.Dataset.from_tensor_slices((x_val_fold,y_val_fold))
            .batch(batch_size, drop_remainder = True)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
  
  model = keras.Sequential(
      [
          keras.Input(shape=(signal_len, 1)),
          import_functions.get_encoder(filters_initial,num_cnn,residual,(signal_len, 1),'ecg_model'),
          layers.Dense(dense_width,kernel_initializer=tf.keras.initializers.HeNormal(seed=1)),
          layers.Dense(num_classes, activation = 'softmax'), #Activation
      ],
      name="model",
  )
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
      )
  num_layers = len(model.layers)-1   
  
  if fold_no == 1:
    model.summary()
    
  history = model.fit(labeled_train_dataset, 
                               epochs = baseline_epochs, 
                               batch_size = batch_size,
                               callbacks = callbacks_list,
                               validation_data = val_dataset,
                               verbose = 2) 
  
  (loss, acc_full) = model.evaluate(test_dataset, verbose=2)
  print(f'[Info]: Val_accu = {acc_full:.4} %')
  Acc_Val_per_fold.append(acc_full)
     
  # Calculating the F1
  F1_2 = import_functions.calculate_f1(model,x_test,y_test)
  F1_per_fold.append(F1_2)
  print('------------------------------------------------------------------------ ')
  print('\n[Info]: Updating Results.txt file')
  print('F1 per fold = '+str(F1_per_fold))
  fold_no = fold_no + 1
  
  print('\n[Info]: Starting model evaluation: ')
  desviacion = np.std(F1_per_fold)
  print('\n[Info]: Standard desviation = '+str(desviacion))
  del(model)

sys.stdout.close()
sys.stdout=stdoutOrigin