from pathlib import Path
import os
import numpy as np
import shutil
import imutils
import cv2
from datetime import datetime
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from Functions.transfer_learning import *
from Duck_Model.freeze_weight import freeze_weight

#Cambios seba
from Functions.query_functions import images_query
from Neural_Network.create_inputs import create_inputs
import json

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

# INPUTS
def all_paths(username = 'Saez', images_folder = 'RF_project', arrays_folder = 'RF_project', model2use='model_1'):
  paths = dict()
  paths['main'] = Path(os.getcwd())
  paths['images'] =  paths['main'] / images_folder
  paths['arrays'] =  paths['main'] / arrays_folder
  result_name = username + '_' + model2use + '_results/' 
  paths['results'] = paths['main'] / result_name   
  paths['file_name'] = ['history_per_train.json', 'score_per_model.json', 'score_per_train.json']
  return paths

def images2arrays(paths, xtr_folder = 'x_tr/images', ytr_folder = 'x_tr/masks', xretr_folder=  'x_retr/images', xtst_folder=None): 
  folders = [xtr_folder, ytr_folder, xretr_folder, xtst_folder]
  ids = dict()

  # Parameters
  im_width  = 512
  im_height = 512
  im_paths  = dict()

  for fold in folders:
    if fold != None:
      im_paths[fold] = paths['images'] / fold
      # List of names all images
      ids[fold]    = next(os.walk(im_paths[fold]))[2]
      print("N° of images in Folder " + fold + " = " + str(len(ids[fold])),"\n")
  
  if ids[xtr_folder] == ids[xtr_folder]:
    x_tr, y_tr = create_inputs(ids[xtr_folder],str(im_paths[xtr_folder]),
                               str(im_paths[ytr_folder]), im_height, im_width)
    
    np.save(paths['arrays'] / 'x_tr.npy', x_tr)
    np.save(paths['arrays'] / 'y_tr.npy', y_tr)

    print('Training arrays were created!')

  if xretr_folder != None:
    x_retr     = create_inputs(ids[xretr_folder],str(im_paths[xretr_folder]),
                               None, im_height, im_width)      
    np.save(paths['arrays'] / 'x_retr.npy',x_retr)
    print('Retraining array was created!')

  if xtst_folder != None:     
    x_tst      = create_inputs(ids[xtst_folder],str(im_paths[xtst_folder]),
                              None, im_height, im_width)
    np.save(paths['arrays'] / 'x_tst.npy',x_tst)           
    print('Test array was created!')          

def load_datasets_from_array(paths, x_tr=True, y_tr=True, x_retr=True, x_tst=True):
  np.random.seed(seed=0)

  # TRAIN SET LABELED
  if x_tr == True and y_tr == True:
    x_tr = np.load(paths['arrays'] / 'x_tr.npy')
    y_tr = np.load(paths['arrays'] / 'y_tr.npy')

    # Split train and validation set
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, 
                                                y_tr, 
                                                test_size = 0.2, 
                                                random_state=42)

    print('Initial x_tr shape', x_tr.shape,'\n')
    print('Initial x_val shape', x_val.shape,'\n')

  # TRAIN SET UNLABELED
  if x_retr == True:
    x_retr = np.load(paths['arrays'] / 'x_retr.npy')
    print('x_retr shape ', x_retr.shape,'\n')

  # TEST SET UNLABELED  
  if x_tst == True:
    x_tst = np.load(paths['arrays'] / 'x_tst.npy')
    print('x_tst shape ', x_tst.shape,'\n')

    return x_tr, y_tr, x_val, y_val, x_retr, x_tst
  
  else: 
    return x_tr, y_tr, x_val, y_val, x_retr

def UNet_engine(paths, model, data, iterations = 3, n_img = 150, model_selected = 'model_1', model_setup=dict()):

  # Inputs
  history_per_train  = dict()
  score_per_model    = dict()
  score_per_train    = dict()
  iterations         = 3                                                              # N° of training
  num_img            = n_img
  xtr                = data['x_tr']
  ytr                = data['y_tr']
  xval               = data['x_val']
  yval               = data['y_val']
  xretr              = data['x_retr']

  create_or_clean_folder(paths)

  # Choose your model
  model_name      = 'model_1'                                                      # Options: Model_i, with i = {1,...,5}
  model_number    = model_name[-1]

  # Iterations
  for iter in range(iterations):

    # Input to query and index
    ui_done = False 

    # Training with labeled data
    print("\n")
    print('Training N° '+ str(iter+1) + ' started')

    model, results =  training_net(model,                                          # model loaded 
                                  model_setup['learning_rate'],                                           # learning rate
                                  'binary_crossentropy',                          # loss function
                                  model_setup['epochs'],                                             # epochs 
                                  model_setup['batch_size'],                                              # batch size
                                  xtr,                                           # x_trian
                                  ytr,                                           # y_train
                                  xval,                                          # x_valid
                                  yval,                                          # y_valid
                                  model_setup['verbose'])                                          # verbose
    history_per_train['Iteration_' + str(iter + 1)] = results

    # save model
    save_model(model,iter+1, paths['results'])

    # Index 
    x_new_tr, xretr = train_images_select(xretr,                                 # x_retrain
                                          int(num_img/iterations),              # N° of images
                                          44)                                     # sedd

    # Prediction
    x_pred    = x_new_tr
    y_pred    = (model.predict(x_pred)>0.5)

    # Compute Score per image and model
    score_per_image = images_query(x_pred, y_pred)
    score_per_train['Iteration_'+str(iter + 1)] = score_per_image
    score_per_model['Iteration_'+str(iter + 1)] = float(np.sum(np.array(score_per_image)>=5))

    # Select images
    x_true, y_true = true_images_select(x_pred, y_pred, score_per_image, threshold = 5)

    xtr, xval, ytr, yval = concatenation_new_dataset(xtr, ytr, x_true, y_true, xval, yval)

    print('Training N° '+ str(iter+1) + ' finished')

  # Save lists
  with open(paths['results']/'history_per_train.json', 'w') as file1:
      json.dump(change_format(history_per_train), file1)
  with open(paths['results']/'score_per_model.json', 'w') as file2:
      json.dump(score_per_model, file2)
  with open(paths['results']/'score_per_train.json', 'w') as file3:
      json.dump(score_per_train, file3)
      

def create_training_test_datasets(params):
  #############################################################
  #
  # INPUTS AND LOAD PARAMETERS
  #
  #############################################################
  now = datetime.now()
  time = now.strftime("%Y-%m-%d %H:%M:%S")

  path       = Path(Path.cwd()) / params['project_name']
  list_img = list((path/params['folder_img']).glob('*.'+params['img_format']))
  create_or_delete_folder(path, 'runs')

  N = params['len_xtr'] + params['len_xretr'] + params['len_xtst']
  img_ix = list(np.random.choice(len(list_img), N, replace=False) )
  X = [list_img[img] for img in img_ix]
  
  xtr   = X[:params['len_xtr']]
  xretr = X[params['len_xtr']:params['len_xtr']+params['len_xretr']]
  xtst  = X[params['len_xtr']+params['len_xretr']:]

  create_or_delete_folder(path, 'x_tr')
  create_or_delete_folder(path, 'x_tr/images')
  create_or_delete_folder(path, 'x_tr/masks')

  save_img_name = open(str(path / 'runs')+"/images_selected_on_"+time+"UTC.txt","w") 

  #############################################################
  #
  # CREATE X_TR FOLDER 
  #
  #############################################################
  save_img_name.write('Training set \n')
  for ix,image in enumerate(xtr):
    print('Creating training folder x_tr... \n')
    print('Processing... image N° '+str(ix)+' of '+str(params['len_xtr']))
    clear_output(wait=True)
    save_img_name.write(str(image)+"\n") 

    if params['processing'] == True:
      # Resize and rotation are applied
      name_img = 'x_tr/images/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      # 512 pixels is used for network requirements
      if params['resize'] == True:
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
      # +90 degrees is according to the horizontal in Counter Clockwise direction
      if params['rotation'] == True:
        img = imutils.rotate(img, angle=params['rotation_angle'])
      # new image is saved
      cv2.imwrite(str(path / name_img), img)
    else:
      name_img = 'x_tr/images/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      cv2.imwrite(str(path / name_img), img)
  save_img_name.write('\n')
  print('Training folder created \n')
  
  #############################################################
  #
  # CREATE X_RETR FOLDER
  #
  #############################################################
  create_or_delete_folder(path, 'x_retr')
  save_img_name.write('Re-Training set \n')
  for ix,image in enumerate(xretr):
    print('Creating training folder x_retr... \n')
    print('Processing... image N° '+str(ix)+' of '+str(params['len_xretr']))
    clear_output(wait=True)
    save_img_name.write(str(image)+"\n") 

    if params['processing'] == True:
      # Resize and rotation are applied
      name_img = 'x_retr/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      # 512 pixels is used for network requirements
      if params['resize'] == True:
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
      # +90 degrees is according to the horizontal in Counter Clockwise direction
      if params['rotation'] == True:
        img = imutils.rotate(img, angle=params['rotation_angle'])
      # new image is saved
      cv2.imwrite(str(path / name_img), img)
    else:
      name_img = 'x_retr/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      cv2.imwrite(str(path / name_img), img)
  save_img_name.write('\n')
  print('Re-training folder created \n')

  #############################################################
  #
  # CREATE TST FOLDER
  #
  #############################################################
  create_or_delete_folder(path, 'x_tst')
  save_img_name.write('Test set \n')
  for ix,image in enumerate(xtst):
    print('Creating training folder x_tst... \n')
    print('Processing... image N° '+str(ix)+' of '+str(params['len_xtst']))
    clear_output(wait=True)
    save_img_name.write(str(image)+"\n") 

    if params['processing'] == True:
      # Resize and rotation are applied
      name_img = 'x_tst/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      # 512 pixels is used for network requirements
      if params['resize'] == True:
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
      # +90 degrees is according to the horizontal in Counter Clockwise direction
      if params['rotation'] == True:
        img = imutils.rotate(img, angle=params['rotation_angle'])
      # new image is saved
      cv2.imwrite(str(path / name_img), img)
    else:
      name_img = 'x_tst/'+str(image)[len(str(path/params['folder_img']))+1:]
      img = cv2.imread(str(image),0)
      cv2.imwrite(str(path / name_img), img)
  save_img_name.write('\n')
    
  print('Test folder created \n')
  
  save_img_name.close() 


def save_img_predicted(paths, masks_ensemble, model_names,folder='x_tst', threshold=0.4):
  data = dict()
  img_name = list((paths['images'] / folder).glob('*.png'))

  for name in model_names:
    create_or_delete_folder(paths['images'], folder+'/'+name)
    data[name] = masks_ensemble[name]
    np.save(paths['images']/ folder / name / 'predicted_masks.npy', data[name])
    
    for ix, image in enumerate(img_name):
      cv2.imwrite(str(paths['images'] / folder / name / img_name[ix]),(data[name][ix].squeeze()>threshold)*255.0 )

  print('Predicted masks saved! ')