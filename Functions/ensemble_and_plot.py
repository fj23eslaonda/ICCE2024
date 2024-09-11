import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json


#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#-------------------------------------------------------------------------------------
def load_results(paths, model_names):
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : load_results
  Description : Load results aim to load results as loss function, 
                f1-score and models scores per iteration
  Inputs      : model_names is a list with model names that you want see the results.
  Date        : 28/10/2021
  """
  
  # dict to save results
  results = dict()
  # Load txt file with results per model
  for model in model_names:
    result  = dict()
    print(model)
    # Read results from txt file
    for file_name in paths['file_name']: 
      file_n       = open(paths['results'] / file_name, 'r') 
      result[file_name] = json.load(file_n) 
    results[model] = result
      
  print('Results loaded')
  return results

#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#-------------------------------------------------------------------------------------
def plot_results_per_model(results, model_names, number_iterations):
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : plot results per model 
  Description : This function is able to plot all results per model and per iterations.
                (a) Loss function per iteration
                (b) F1-Score per iteration
                (c) Score trian per iteration
  Inputs      : model_names is a list with model names that you want see
                results is a dict with results per model
                number_iterations is a list with iteration number that you want see
  Date        : 28/10/2021
  """
  for model in model_names:
    model_names = [model]
    fig, axs = plt.subplots(1, 3,figsize=(18,5))
    fig.suptitle('Model '+ model[-1] + ' results', fontsize=20,fontweight="bold")
    parameters = {'axes.labelsize': 14,
                  'axes.titlesize': 17,
                  'legend.fontsize': 12,
                  'xtick.labelsize': 13,
                  'ytick.labelsize': 13}
    plt.rcParams.update(parameters)
    color = ['r','b','g']
    for n_model in model_names:
      print(n_model)
      width = 0.3
      if len(results[n_model]) == 1:
        start = 0
      elif len(results[n_model])==2:
        start = 0
      else:
        start = -width  
        
      for n_iter in number_iterations:
        n_iteration = int(n_iter[-1:])-1
        # Loss Function to iteration 1
        axs[0].plot(results[n_model]['history_per_train.json'][n_iter]['loss'], color = color[n_iteration], label = n_iter + '- Train')
        axs[0].plot(results[n_model]['history_per_train.json'][n_iter]['val_loss'], color = color[n_iteration], linestyle='--', label = n_iter + '- Val')
        axs[0].set_title('Loss Function')
        axs[0].set_xlabel('N Epochs')
        axs[0].set_ylabel('Binary Cross-Entropy')
        axs[0].set_ylim(0,1)
        axs[0].legend(loc='best')

        ## F1 Function to iteration 1
        axs[1].plot(results[n_model]['history_per_train.json'][n_iter]['f1'], color = color[n_iteration], label = n_iter + '- Train')
        axs[1].plot(results[n_model]['history_per_train.json'][n_iter]['val_f1'], color = color[n_iteration], linestyle='--', label = n_iter + '- Val')
        axs[1].set_title('F1-Score')
        axs[1].set_xlabel('N Epochs')
        axs[1].set_ylabel('F1-Score')
        axs[1].set_ylim(0,1)
        axs[1].legend(loc='best')

        # Bars Graphs
        axs[2].set_title('Score Bar Graph')
        axs[2].set_xlabel('Score')
        axs[2].set_ylabel('N images')
        bars_key = list()
        bars_values = list()
        histo = dict(zip(results[n_model]['score_per_train.json'][n_iter],map(lambda x: results[n_model]['score_per_train.json'][n_iter].count(x),results[n_model]['score_per_train.json'][n_iter])))
        keys = list(histo.keys())
        values = list(histo.values())
        axs[2].bar([key+start for key in keys], values, width,color=color[n_iteration], label = n_iter)
        start+=width
        axs[2].legend(loc='best')
          
      plt.show()
      
#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#-------------------------------------------------------------------------------------
def load_models(paths, model_names, number_iterations): 
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : plot results per model 
  Description : This function is able to plot all results per model and per iterations.
                (a) Loss function per iteration
                (b) F1-Score per iteration
                (c) Score trian per iteration
  Inputs      : model_names is a list with model names that you want see
                results is a dict with results per model
                number_iterations is a list with iteration number that you want see
  Date        : 28/10/2021
  """
  models_loaded = dict()
  for model_n in model_names:

    for n_iter in number_iterations:
      iter = n_iter[-1]

      # Load JSON and Create model
      model_name         =  'model_'+iter+'.json'
      json_file          =  open(paths['results']/ model_name, 'r')
      loaded_model_json  =  json_file.read()
      json_file.close()
      model              = model_from_json(loaded_model_json, {"tf":tf})

      # Load weight 
      weight_name = 'best_model_'+iter+'.h5'
      model.load_weights(str(paths['results'] / weight_name))
      
      models_loaded[model_n+'/'+n_iter] = model
      print(model_n + ' ' + 'of ' + n_iter+ ' Loaded')
  return models_loaded
  
#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#-------------------------------------------------------------------------------------

def load_score_per_models(results, model_names, number_iterations):
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : plot results per model 
  Description : This function is able to plot all results per model and per iterations.
                (a) Loss function per iteration
                (b) F1-Score per iteration
                (c) Score trian per iteration
  Inputs      : model_names is a list with model names that you want see
                results is a dict with results per model
                number_iterations is a list with iteration number that you want see
  Date        : 28/10/2021
  """
  score_per_model = dict()
  for model_n in model_names: 
    score = results[model_n]['score_per_model.json']
    for n_iter in number_iterations:
      score_per_model[model_n+'/'+n_iter] = score[n_iter]
  print('Scores Loaded')
  return score_per_model

#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#-------------------------------------------------------------------------------------

def ensemble_models(models_loaded, models_to_ensemble, scores_loaded, number_iterations, x_tst):
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : plot results per model 
  Description : This function is able to plot all results per model and per iterations.
                (a) Loss function per iteration
                (b) F1-Score per iteration
                (c) Score trian per iteration
  Inputs      : model_names is a list with model names that you want see
                results is a dict with results per model
                number_iterations is a list with iteration number that you want see
  Date        : 28/10/2021
  """
  mask_ensemble   = dict()
  y_ensemble      = np.zeros((len(x_tst),512,512,1))
  score           = list()

  for model_n in models_to_ensemble:
    for n_iter in number_iterations:
      score.append(scores_loaded[model_n+'/'+n_iter])

  sum_score = np.sum(score)
  for model_n in models_to_ensemble:
    for n_iter in number_iterations:
      score_per_model = scores_loaded[model_n+'/'+n_iter]/sum_score
      model      = models_loaded[model_n+'/'+n_iter]
      y_pred     = model.predict(x_tst, verbose=True)
      y_ensemble = y_ensemble + y_pred*score_per_model
      del model, y_pred
      print( )
      print(model_n +' of '+ n_iter + ' prediction done with weight= ' +str(np.round(score_per_model,2)*100)+'%')
      print( )
    mask_ensemble[model_n] = y_ensemble
  
  return mask_ensemble

#-------------------------------------------------------------------------------------
#
# LOAD DATA
#
#------------------------------------------------------------------------------------- 
  
def plot_predictions(x_tst, masks_ensemble, model_to_ensemble, index = 159, threshold=0.2):
  """"
  Project     : Wave-by-Wave breaking patterns identification
  Function    : plot results per model 
  Description : This function is able to plot all results per model and per iterations.
                (a) Loss function per iteration
                (b) F1-Score per iteration
                (c) Score trian per iteration
  Inputs      : model_names is a list with model names that you want see
                results is a dict with results per model
                number_iterations is a list with iteration number that you want see
  Date        : 28/10/2021
  """
  fig, ax = plt.subplots(1, 4, figsize = (24,10))

  ix = index
  threshold = threshold

  # Figure 1
  ax[0].imshow(x_tst[ix].squeeze(), cmap = 'gray')
  ax[0].grid(False)
  ax[0].set_title('Video image')
  ax[0].set_xlabel('Cross-shore Distance [pixel]')
  ax[0].set_xlabel('Alongshore Distance [pixel]')

  # Figure 2
  ax[1].imshow(masks_ensemble[model_to_ensemble[0]][ix].squeeze(), cmap = 'gray')
  ax[1].grid(False)
  ax[1].set_title('Prediction mask')
  ax[1].set_xlabel('Cross-shore Distance [pixel]')

  # Figure 3
  ax[2].imshow((masks_ensemble[model_to_ensemble[0]][ix]>threshold).squeeze(), cmap = 'gray')
  ax[2].grid(False)
  ax[2].set_title('Prediction Binary mask')
  ax[2].set_xlabel('Cross-shore Distance [pixel]')

  # Figure 4
  ax[3].imshow(x_tst[ix].squeeze(), cmap = 'gray')
  ax[3].contour((masks_ensemble[model_to_ensemble[0]][ix]>threshold).squeeze(), colors = 'r', levels = [0.1])
  ax[3].grid(False)
  ax[3].set_title('Prediction Binary mask')
  ax[3].set_xlabel('Cross-shore Distance [pixel]')

  plt.show()

def create_or_delete_folder(path, folder):
    
  # getting the folder path from the user
  
  folder_path = str(path / folder)

  if os.path.exists(folder_path):
    # DELETE RESULT FOLDER AND ITS CONTENTS
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    (path / folder).mkdir(mode=0o755, exist_ok=True)
  else:
    (path / folder).mkdir(mode=0o755, exist_ok=True)


def save_img_predicted(paths, masks_ensemble, model_names,folder='x_tst', threshold=0.4):
  data = dict()
  img_name = os.listdir(paths['images'] / folder)

  for name in model_names:
    create_or_delete_folder(paths['images'], folder+'/'+name)
    data[name] = masks_ensemble[name]
    np.save(paths['images']/ folder / name / 'predicted_masks.npy', data[name])
    
    for ix, image in enumerate(img_name):
      cv2.imwrite(str(paths['images'] / folder / name / img_name[ix]),(data[name][ix].squeeze()>threshold)*255.0 )

  print('Predicted masks saved! ')