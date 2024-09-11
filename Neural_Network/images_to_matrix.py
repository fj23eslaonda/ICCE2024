import os
from Neural_Network.create_inputs import *


def images_to_matrix(image_tr, label_tr, im_retr,name_im_tr, name_la_tr, name_im_retr):
    # Parameters
    im_width  = 512
    im_height = 512

    # IF YOU WANT TO TRANSFORM JUST TRAIN AND RETRAIN IMAGES

    if label_tr == None:
        paths     = {"im_tr":image_tr, #"la_tr":"data_cruces/label/", 
                     "im_retr":im_retr}

        # List of names all images
        ids_tr    = next(os.walk(paths["im_tr"]))[2]
        ids_retr  = next(os.walk(paths["im_retr"]))[2]

        print("N° of images in train set = ", len(ids_tr),"\n")
        print("N° of images in second train set = ", len(ids_retr),"\n")

        # Creating of matrix for train and test set

        x_tr   = create_inputs(ids_tr,paths["im_tr"],
                                   None, im_height, im_width) 

        x_retr    = create_inputs(ids_retr,paths["im_retr"],
                                   None, im_height, im_width)  

        # Save inputs
        np.save(name_im_tr,x_tr)
        np.save(name_im_retr,x_retr)

        

    # IF YOU WANT TO TRANSFORM JUST IMAGES AND LABEL TRAIN
    elif im_retr == None:
        paths     = {"im_tr":image_tr, "la_tr":label_tr, 
                     #"im_retr": im_retr
                     }

        # List of names all images
        ids_tr       = next(os.walk(paths["im_tr"]))[2]
        ids_tr_la    = next(os.walk(paths["la_tr"]))[2]

        print("N° of images in train set = ", len(ids_tr),"\n")
        print("N° of labels in train set = ", len(ids_tr_la),"\n")

        # Creating of matrix for train and test set

        x_tr,y_tr   = create_inputs(ids_tr,paths["im_tr"],
                               paths["la_tr"], im_height, im_width) 

        # Save inputs
        np.save(name_im_tr,x_tr)
        np.save(name_la_tr,y_tr)


    # IF YOU WANT TO TRANSFORM TRAIN AND RETRAIN IMAGES AND LABELS TRAIN
    else:
        paths     = {"im_tr":image_tr, "la_tr":label_tr, 
                     "im_retr": im_retr}

        # List of names all images
        ids_tr       = next(os.walk(paths["im_tr"]))[2]
        ids_tr_la    = next(os.walk(paths["la_tr"]))[2]
        ids_retr     = next(os.walk(paths["im_retr"]))[2]

        print("N° of images in train set = ", len(ids_tr),"\n")
        print("N° of labels in train set = ", len(ids_tr_la),"\n")
        print("N° of images in second train set = ", len(ids_retr),"\n")

        # Creating of matrix for train and test set

        x_tr,y_tr   = create_inputs(ids_tr,paths["im_tr"],
                               paths["la_tr"], im_height, im_width) 

        x_retr    = create_inputs(ids_retr,paths["im_retr"],
                                   None, im_height, im_width)  

        # Save inputs
        np.save(name_im_tr,x_tr)
        np.save(name_la_tr,y_tr)
        np.save(name_im_retr,x_retr)
        
