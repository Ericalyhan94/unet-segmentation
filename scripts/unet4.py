import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import cv2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
import shutil
from tensorflow.keras.layers import concatenate
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"
import numpy as np
import pandas as pd 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import math

merge_num=22


 
################### 4.Split Your Dataset into Unmasked and Masked Imagesages ###########################





  
start = timer()
test3 = unet.evaluate(processed_image_ds_test.cache().shuffle(BUFFER_SIZE).batch(1))
end = timer()
print('Time Used For Test Set Evaludation:',end - start)

print('#################################################')
print('after fine tuning: Images from the Web')
print('test set loss:' + str(test3[0]))
print('test set accuracy:' + str(test3[1]))


size_train=len(processed_image_ds_train)
size_validation=len(processed_image_ds_validation)
size_test=len(processed_image_ds_test)
print('Train size: ',size_train,'Validation size: ',size_validation,'Test size: ',size_test)


###########################Save Model Predicted Images ###############################################################

 
home_dir=os.getcwd()
path_train = r'{}/Predictions/train/'.format(home_dir) 
path_validation = r'{}/Predictions/validation/'.format(home_dir) 
path_test = r'{}/Predictions/test/'.format(home_dir) 
path_all = r'{}/Predictions/all/'.format(home_dir) 

def save_resized_images_to_folder(figures,saved_path):

    if  os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    os.chdir(saved_path)


    for i in figures.keys():
        figures[i]=figures[i].astype(np.float32)
        figures[i] = cv2.resize(figures[i], (640,480), interpolation = cv2.INTER_AREA)        
        figures[i]  = cv2.cvtColor(figures[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite('{}.jpg'.format(i), figures[i])
       


def save_predictions(dataset=None, num=1,name='train'):  
    figures={}
    mm=0  
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            pred_mask = manipulation_mask(create_mask(pred_mask))
            figures[name+str(mm)]=pred_mask
            mm+=1
        return figures
    else:
        print('Dataset is Empty!')
    

train_display=processed_image_ds_train.cache().shuffle(BUFFER_SIZE).batch(1)
validation_display=processed_image_ds_validation.cache().shuffle(BUFFER_SIZE).batch(1)
test_display=processed_image_ds_test.cache().shuffle(BUFFER_SIZE).batch(1)


figures_train=save_predictions(train_display, len(train_display),name='train')
figures_validation=save_predictions(validation_display, len(validation_display),name='validation')
figures_test=save_predictions(test_display, len(test_display),name='test')
figures_all={**figures_train,**figures_validation,**figures_test}

save_resized_images_to_folder(figures_train,path_train)
save_resized_images_to_folder(figures_validation,path_validation)
save_resized_images_to_folder(figures_test,path_test)
save_resized_images_to_folder(figures_all,path_all)


######################Superposition of Figures ##########################################################


path_train_superposition = r'{}/Predictions/train_superposition/'.format(home_dir) 
path_validation_superposition = r'{}/Predictions/validation_superposition/'.format(home_dir) 
path_test_superposition = r'{}/Predictions/test_superposition/'.format(home_dir) 


def save_predictions_superposition(dataset=None, num=1,name='train'):  
    figures={}
    mm=0  
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            pred_mask = manipulation_mask(create_mask2(pred_mask,mask)).astype(float)
            figures[name+str(mm)]=pred_mask
            mm+=1
        return figures
    else:
        print('Dataset is Empty!')


figures_train_superposition=save_predictions_superposition(train_display, len(train_display),name='train')
figures_validation_superposition=save_predictions_superposition(validation_display, len(validation_display),name='validation')
figures_test_superposition=save_predictions_superposition(test_display, len(test_display),name='test')

save_resized_images_to_folder(figures_train_superposition,path_train_superposition)
save_resized_images_to_folder(figures_validation_superposition,path_validation_superposition)
save_resized_images_to_folder(figures_test_superposition,path_test_superposition)


#######################################################################################################################
#####This part is to read bainite and martensite and transform data into dataset for predictions and comparisons#######
#######################################################################################################################

bainite_mask_list=os.listdir(bainite_mask_path_train)
bainite_image_list=os.listdir(bainite_image_path_train)
martensite_mask_list=os.listdir(martensite_mask_path_train)
martensite_image_list=os.listdir(martensite_image_path_train)
 

bainite_mask_list = [bainite_mask_path_train+i for i in bainite_mask_list]
bainite_image_list = [bainite_image_path_train+i for i in bainite_image_list]
martensite_mask_list= [martensite_mask_path_train+i for i in martensite_mask_list]
martensite_image_list= [martensite_image_path_train+i for i in martensite_image_list]

bainite_mask_list.sort()
bainite_image_list.sort()
martensite_mask_list.sort()
martensite_image_list.sort()

mask_list_ds_bainite = tf.data.Dataset.list_files(bainite_mask_list, shuffle=False)
image_list_ds_bainite= tf.data.Dataset.list_files(bainite_image_list, shuffle=False)
mask_list_ds_martensite = tf.data.Dataset.list_files(martensite_mask_list, shuffle=False)
image_list_ds_martensite = tf.data.Dataset.list_files(martensite_image_list, shuffle=False)

 
masks_filenames_bainite = tf.constant(bainite_mask_list)
image_filenames_bainite = tf.constant(bainite_image_list)
masks_filenames_martensite = tf.constant(martensite_mask_list)
image_filenames_martensite = tf.constant(martensite_image_list)

 
dataset_bainite = tf.data.Dataset.from_tensor_slices((image_filenames_bainite, masks_filenames_bainite))
dataset_martensite = tf.data.Dataset.from_tensor_slices((image_filenames_martensite , masks_filenames_martensite))


image_ds_bainite = dataset_bainite.map(process_path)
image_ds_martensite = dataset_martensite.map(process_path)

processed_image_ds_bainite = image_ds_bainite.map(preprocess)
processed_image_ds_martensite = image_ds_martensite.map(preprocess)


bainite_dataset = processed_image_ds_bainite.cache().shuffle(BUFFER_SIZE).batch(1)
martensite_dataset= processed_image_ds_martensite.cache().shuffle(BUFFER_SIZE).batch(1)

###################       Save Bainite and Martensite   ####################


path_bainite = r'{}/Predictions/bainite/'.format(home_dir) 
path_martensite = r'{}/Predictions/martensite/'.format(home_dir) 

figures_bainite=save_predictions(bainite_dataset, len(bainite_dataset),name='bainite')
figures_martensite =save_predictions(martensite_dataset, len(martensite_dataset),name='martensite')

save_resized_images_to_folder(figures_bainite,path_bainite)
save_resized_images_to_folder(figures_martensite,path_martensite)



###########################################################################################################
#####This part is for predicting pictures together and work out the size distributions of precipitates #####
###########################################################################################################


def save_images_for_size(file_folder,save_path):
    if  os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for filename in os.listdir(file_folder):
        print(filename)
        if filename[-3:] in ['png','jpg','gif','svg','peg']:
            img = imageio.imread(os.path.join(file_folder,filename))
            img = tf.image.resize(img, (96, 128), method='nearest')
            pred_img = unet.predict(img[None,:,:])
            pred_img = pred_img.astype(np.float32)
            pred_img = np.array(pred_img)
            pred_img = manipulation_mask(create_mask(pred_img))
            pred_img = cv2.resize(pred_img, (640,480), interpolation = cv2.INTER_AREA)  
        
            if pred_img is not None:
                pred_img = pred_img.astype(np.float32)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite('{}{}'.format(save_path,filename), pred_img)

path_bainite_forsize = r'{}/Predictions/bainite_forsize/'.format(home_dir) 
path_martensite_forsize = r'{}/Predictions/martensite_forsize/'.format(home_dir) 





save_images_for_size(lower_bainite_for_size,path_bainite_forsize)
save_images_for_size(tempered_martensite_for_size,path_martensite_forsize)
