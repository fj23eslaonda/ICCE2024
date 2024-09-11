import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from Neural_Network.f1 import f1

def true_images_select(x_pred, y_pred,score_per_image,threshold):

  score = np.asarray(score_per_image) >= threshold

  x_true, y_true     =  [], [], 

  for n_image in range(len(x_pred)):  

    if score[n_image] == True:
      x_true.append(x_pred[n_image])
      y_true.append(y_pred[n_image])
    

  return np.asarray(x_true), np.asarray(y_true)

def train_images_select(x_retr, count, seed ):
  np.random.seed(seed=seed)
  numbers = np.random.choice(range(len(x_retr)), count, replace=False)
  x_tr    = x_retr[numbers]
  x_retr  = np.delete(x_retr, numbers, axis = 0)

  return x_tr, x_retr
  

def training_net(model,lear_rate,loss_func , n_epochs , n_batchs ,x_tr, y_tr, x_val, y_val, verbose ):

  # Inputs
  input_img = Input((512, 512, 1), name = 'img')

  # Compilate
  #model = get_unet(input_img, n_filters = 32, dropout = 0.2, batchnorm = True, seed = 0)
  model.compile(optimizer= Adam(learning_rate=lear_rate),loss=loss_func, metrics = [f1], run_eagerly=True)

  # Overfitting solution
  callbacks = [
  EarlyStopping(patience = int(n_epochs/5), verbose = verbose),
  ReduceLROnPlateau(factor = 0.1, patience = int(n_epochs/10), min_lr = 0.00001, verbose = verbose),
  ModelCheckpoint('best_model.h5', verbose = verbose, save_best_only = True, save_weights_only = True)
  ]

  # Fit net
  unet = model.fit(x_tr,y_tr,batch_size= n_batchs, epochs = n_epochs, verbose = verbose, callbacks = callbacks,
                 validation_data=(x_val,y_val))
  
  # Save results

  return model, unet.history
  
  
def save_model(model, index, path):
  # serializar el modelo a JSON
  model_json = model.to_json()
  model_name = "model_"+str(index)+".json"
  with open(path / model_name, "w") as json_file:
      json_file.write(model_json)
  # serializar los pesos a HDF5
  weight_name = "best_model_"+str(index)+".h5"
  model.save_weights(path / weight_name)
  print(" ")
  print("Model saved")
  
def concatenation_new_dataset(x_tr, y_tr, x_true, y_true, x_val, y_val):
    # Concatenate
    x_tr = np.concatenate((x_tr, x_val), axis = 0)
    y_tr = np.concatenate((y_tr, y_val), axis = 0)
    x_tr = np.concatenate((x_tr, x_true), axis = 0)
    y_tr = np.concatenate((y_tr, y_true), axis = 0)
    
    # Create new training set  
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size = 0.2, random_state=42)
    
    return x_tr, x_val, y_tr, y_val

def create_or_clean_folder(paths):
  path_results = paths['results']
  if os.path.exists(path_results):
    files = os.listdir(path_results)
    for file in files:
      os.remove(f'{path_results}/{file}')
  else: 
    os.mkdir(path_results)
    
    
def change_format(history_per_train):
    for iter in list(history_per_train.keys()):
        for metric in list(history_per_train[iter].keys()):
            values = history_per_train[iter][metric]
            values = [float(np.round(x, 5)) for x in values]
            history_per_train[iter][metric] = values    
    return history_per_train
