#from Functions.all_packages import *
from Functions.friendly_functions import *
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from Functions.ensemble_and_plot import ensemble_models 

def full_dataset_processing(params, img_from):
  #############################################################
  #
  # INPUTS AND LOAD PARAMETERS
  #
  #############################################################

  path       = Path(Path.cwd()) / params['project_name']
  # -----------------------------------------------------------
  list_img   = (path/img_from).glob('*.'+params['img_format'])
  list_img   = sorted([str(img_path) for img_path in list_img])
  list_img   = [Path(img_path) for img_path in list_img]
  # -----------------------------------------------------------
  img_in_parts = [list_img[x:x+400] for x in range(0, len(list_img), 400)]
  # -----------------------------------------------------------
  new_folder   = img_from + '_predicted'
  create_or_delete_folder(path, new_folder)

  # -----------------------------------------------------------
  for jx, subset in enumerate(img_in_parts):
    img_post = list()
    # -----------------------------------------------------------
    name_txt = img_from+'_subset_'+str(jx+1)+'.txt'
    with open(path / new_folder/ name_txt, 'w') as output:
      for row in subset:
          output.write(str(row) + '\n')

    # -----------------------------------------------------------
    for ix,image in enumerate(subset):
      print('Creating your test set ... \n')
      print('Processing... image N° '+str(ix)+' of '+str(len(subset))+ ' - Subset '+ str(jx+1))
      clear_output(wait=True)
      # -----------------------------------------------------------
      if params['processing'] == True:
        # Resize and rotation are applied
        print(image)
        name_img = str(image)[len(str(path/img_from))+1:]
        img = cv2.imread(str(image),0)
        # 512 pixels is used for network requirements
        if params['resize'] == True:
          img = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        # +90 degrees is according to the horizontal in Counter Clockwise direction
        if params['rotation'] == True:
          img = imutils.rotate(img, angle=params['rotation_angle'])
        # new image is saved
        img_post.append(img)
      # -----------------------------------------------------------
      else:
        name_img = str(image)[len(str(path/img_from))+1:]
        img = cv2.imread(str(image),0)
        img_post.append(img)
    # -----------------------------------------------------------
    img_post = np.asarray(img_post)
    name_npy = img_from+'_subset_'+str(jx+1)+'.npy'
    np.save(path / new_folder/ name_npy, img_post)
    print('Subset N°'+str(jx+1)+' of '+str(len(img_in_parts))+' was created \n')

    # dividido el conjunto principal, falta hacer la predicción sobre ese conjunto
    # y guardar las máscaras con el mismo nombre
    

def print_message(message):
  print('#----------------------------------------------')
  print('#')
  print('#      '+ message)
  print('#')
  print('#----------------------------------------------')
  print('\n')

def prediction_on_whole_dataset(params, paths, img_from, model_names, models_loaded,scores_loaded,number_iterations):

  # -----------------------------------------------------------
  path        = Path(Path.cwd()) / params['project_name']
  folder      = img_from +'_predicted'
  list_arrays = list((path/folder).glob('*.npy'))
  list_txt    = list((path/folder).glob('*.txt'))
  
  # -----------------------------------------------------------
  for name in model_names:
    jx = 1
    # -----------------------------------------------------------
    for array, txt in zip(list_arrays, list_txt):
      message = name + ' predictions - subset N°'+ str(jx)
      print_message(message)

      x_tst = np.load(array)
      x_tst = x_tst/255.0
      x_tst = x_tst[..., np.newaxis]

      # -----------------------------------------------------------
      with open(txt) as f:
        img_path = f.readlines()
        # 42 is the length of images name
        img_name = [name[-params['len_img_name']:] for name in img_path]

      # -----------------------------------------------------------

      # Ensemble process
      masks_ensemble = ensemble_models(models_loaded,                                 # Models to ensemble
                                      model_names,                                    # Model to use
                                      scores_loaded,                                  # Score per model
                                      number_iterations,                              # Iterations per model
                                      x_tst)                                          # Test Set
      del x_tst
      save_matrix_predicted(paths, params, masks_ensemble, name, img_from, jx)
      jx+=1
      

def save_matrix_predicted(paths, params, masks_ensemble, name, img_from, jx):
  data = dict()
  path = Path(Path.cwd()) / params['project_name']

  # -----------------------------------------------------------
  folder_name = 'Predictions_'+img_from+'_'+name
  create_or_delete_folder(paths['images'],folder_name)

  # -----------------------------------------------------------
  new_folder = img_from+'_predicted'
  name_masks = 'predicted_masks_'+str(jx)+'.npy'

  # -----------------------------------------------------------
  np.save(path / new_folder  / name_masks, masks_ensemble[name])

  print('Subset N°'+ str(jx)+': masks saved! \n ')

def save_img_predicted(paths, params, name, img_from, threshold):
  # -----------------------------------------------------------
  path        = Path(Path.cwd()) / params['project_name']
  folder_name  = 'Predictions_'+img_from + '_' + name
  array_folder = img_from + '_predicted'
  list_arrays = list((path/array_folder).glob('predicted*'))
  list_txt    = list((path/array_folder).glob('*.txt'))
  # -----------------------------------------------------------
  for array, txt in zip(list_arrays, list_txt):
    masks_ensemble = np.load(array)
    # -----------------------------------------------------------
    with open(txt) as f:
      img_path = f.readlines()
      img_name = [name[-(params['len_img_name']+1):-1] for name in img_path]

    for ix in tqdm(range(len(img_name))):
      mask = Image.fromarray((masks_ensemble[ix].squeeze()>=threshold)*255.0)
      mask = mask.convert("L")
      mask.save(str(path / folder_name / img_name[ix]))