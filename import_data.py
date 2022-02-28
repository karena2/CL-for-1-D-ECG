import numpy as np 
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import random
import tensorflow.keras as keras


## Import  data 
def load_data_train(length, signal_len, samples = 0): 

    if length == "double":
      dataset = h5py.File('/home/kfonseca_cps/dataset_doble/datasetd_05_2', 'r') 
      flag = 0
        
    elif length == "simple":
      flag = 1
      dataset = h5py.File('/home/kfonseca_cps/datasets_3/train_025_3.h5', 'r') 
      
    data = dataset.get('data')
    data = np.array(data)
    labels = dataset.get('labels')
    labels = np.array(labels)
    dataset.close()
    
    # Depends on what dataset is, we have to ensure that there are not the same signals
    
    if flag == 1:
      data_to_export, _, labels_to_export, _ = train_test_split(data,labels, train_size=samples/labels.shape[0], random_state=0, stratify=labels)    
      classes = tf.math.bincount(labels_to_export)
      classes = np.array(classes)
      print('[Info]: Data')
      print('Number of train samples: ',data_to_export.shape[0])
      print('Distribution of classes: ',classes)   
       
    elif flag == 0:
      p1 = np.random.permutation(len(data))
      data_to_export = data[p1]
      labels_to_export = labels[p1]
      del(data)
      del(labels)
    
    data_to_export = data_to_export[:,0:signal_len,:] 
      
    return data_to_export, labels_to_export
    
## Import test function
def load_data_test(signal_len): 
    dataset = h5py.File('/home/kfonseca_cps/datasets_3/test_3', 'r')
    data = dataset.get('data')
    data = np.array(data)
    labels = dataset.get('labels')
    labels = np.array(labels)
    dataset.close()
    classes = tf.math.bincount(labels)
    classes = np.array(classes)
    print('Number of test samples: ',data.shape[0])
    print('Distribution of classes: ',classes)
    data = data[:,0:signal_len,:]
    return data, labels    

## Process pre-training data 
def preprocess_data_time_division(data,batch_size,batches,val_batches):

  """
  Builds PCLR data: number of batches x twice number of patients(two signals per patient)
  Parameters
  ----------
  data,         ecg signals of patients
  batch_size,   size of the batch  
  """    
  half = int(data.shape[0]/2)
  data1 = data[0:half,:]
  data2 = data[half:,:]
  val = []
  train = []
  
  # pre-training data
  for i in range(batches):
    first_half_batch = []
    second_half_batch = []
    for j in range(0,batch_size):
      doble = data1[j+i*batch_size,:]
      doble = doble[0:4096]
      first_ecg, second_ecg = tf.split(doble, 2, 0)
      noise_1 = np.random.normal(0,0.03, first_ecg.shape)
      noise_2 = np.random.normal(0,0.05, second_ecg.shape)
      first_ecg = first_ecg + noise_1
      second_ecg = second_ecg + noise_2
      first_half_batch.append(first_ecg)
      second_half_batch.append(second_ecg)
      
    train.append(np.concatenate((first_half_batch,second_half_batch),axis=0))
    
  # validation data
  for i in range(val_batches):
    first_half_batch = []
    second_half_batch = []
    for j in range(0,batch_size):
      doble = data2[j+i*batch_size,:]
      doble = doble[0:4096]
      first_ecg, second_ecg = tf.split(doble, 2, 0)
      noise_1 = np.random.normal(0,0.03, first_ecg.shape)
      noise_2 = np.random.normal(0,0.05, second_ecg.shape)
      first_ecg = first_ecg + noise_1
      second_ecg = second_ecg + noise_2
      first_half_batch.append(first_ecg)
      second_half_batch.append(second_ecg)
    val.append(np.concatenate((first_half_batch,second_half_batch),axis=0))
    
  del(data1)
  del(data2)
  train = np.concatenate(train,axis=0)
  val = np.concatenate(val,axis=0)
  
  return train,val