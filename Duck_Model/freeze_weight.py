from pathlib import Path
import tensorflow as tf
from keras.models import Model, load_model, model_from_json


def freeze_weight(paths,option):
    
    # Load JSON and Create model
    json_file          =  open(paths['main'] / 'Duck_Model' / 'model_final_V2.json', 'r')
    loaded_model_json  =  json_file.read()
    json_file.close()
    model              = model_from_json(loaded_model_json, {"tf":tf})

    # Load weight 
    model.load_weights(paths['main'] / 'Duck_Model' / 'best_model_final_V2.h5')

    if option == 'model_1':

        i = 0
        for layers in model.layers:
          if i<= 9:
            layers.trainable = False
          i+=1

    elif option == 'model_2':

        i = 0
        for layers in model.layers:
          if i<= 23:
            layers.trainable = False
          i+=1

    elif option == 'model_3':

        i = 0
        for layers in model.layers:
          if i>= 36:
            layers.trainable = False
          i+=1
          
    elif option == 'model_4':

        i = 0
        for layers in model.layers:
          if i>= 24:
            layers.trainable = False
          i+=1

    else:
        
        i = 0
        for layers in model.layers:
          if i<= 9 or i >= 36:
            layers.trainable = False
          i+=1

    print("Model Loaded \n")

    return model


