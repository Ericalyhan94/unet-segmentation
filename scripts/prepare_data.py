## Please put this file in the directory above martensite or bainite directory 
## This figure will get into the bainite or martensite directory and create resized images /cropped images in the directory automatically
## Xiaohan Bie 2022-07-13


##
import PIL
from PIL import Image
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
import cv2
from sklearn.model_selection import train_test_split

## All the images are resized to 10000 magnification. You can change this value accordingly
desired_magnification=10000
height = 480
weight = 640
desired_magnification=int(desired_magnification)


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
data_dir = os.path.abspath(data_dir)

base_dir=data_dir
train_dir = os.path.join(base_dir, 'training_set')
validation_dir = os.path.join(base_dir, 'validation_set')
test_dir = os.path.join(base_dir, 'test_set')
current_path = data_dir


#this function loads all the images from a specified folder. change the size, and output in a dictionary
def load_images_from_folder(folder,type=None,write=False):
    if write==True:
        output_file=os.path.join(current_path,"{}_without_resize.txt".format(type))
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'w') as f:
            f.write('')

    images = {}
    for filename in os.listdir(folder):
        if filename[-3:] in ['png','jpg','gif','svg','peg']:
            magnification=(int(filename.split('X')[0].split('-')[1]))
            change_ratio=desired_magnification/magnification
            img = cv.imread(os.path.join(folder,filename))
            [y_dim,x_dim]=img.shape[:2]
            if write==True:
                print('Output Files: ',output_file)
                with open(output_file,"a") as f:  
                    print('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(filename).split('.png')[0],y_dim,x_dim))
                    f.write('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(filename).split('.png')[0],y_dim,x_dim))        
       
            print('xdim:',x_dim,'y_dim',y_dim)
            x_dim=int(x_dim*change_ratio)
            y_dim=int(y_dim*change_ratio)
            img = cv.resize(img,(x_dim,y_dim))
            [x_dim,y_dim]=img.shape[:2]
            print('redized_xdim:',x_dim,'redized_ydim',y_dim)
 
            if img is not None:
                images[str(filename)]=(img)
    return images
	
	
def save_resized_images_to_folder(images,saved_path):
    if  os.path.exists(saved_path):
        shutil.rmtree(saved_path)
    os.mkdir(saved_path)
    os.chdir(saved_path)
    for i in images.keys():
        print(images[i].shape)
        plt.imshow(images[i])
        plt.imsave(i,images[i])
        plt.show()
		
		
def gray(img):
  return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
		
#images are defined in a dictionary
def cropSave(images, type,h, w, saved_path,write=False):
    if write==True:
        output_file=os.path.join(current_path,"{}_output.txt".format(type))
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'w') as f:
            f.write('')
    if  os.path.exists(saved_path):
        shutil.rmtree(saved_path)  
    os.mkdir(saved_path)
    os.chdir(saved_path)
    
    for i in images.keys():
        [y_dim,x_dim]=images[i].shape[:2]
        if write==True:
            print('Output Files: ',output_file)
            with open(output_file,"a") as f:  
                print('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(i).split('.png')[0],y_dim,x_dim))
                f.write('Name of Images: {} , Shape of Images:y and x dimension: {} {}\n'.format(str(i).split('.png')[0],y_dim,x_dim))        
        for y in range(int(np.ceil(y_dim / h))):
            for x in range(int(np.ceil(x_dim/ w))):
                if (y ==np.ceil(y_dim / h)-1) and (x ==np.ceil(x_dim / w)-1):
                    print('The last y is:',y,'The last x is :',x)
                    cropped_img = images[i][images[i].shape[0]-h:images[i].shape[0],  images[i].shape[1]-w:images[i].shape[1]]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                if y ==np.ceil(y_dim / h)-1:
                    cropped_img = images[i][images[i].shape[0]-h:images[i].shape[0], x*w:(x+1)*w]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                if x ==np.ceil(x_dim / w)-1:
                    cropped_img = images[i][y*h:(y+1)*h, images[i].shape[1]-w:images[i].shape[1]]
                    image_name=(type+str(i) + str(y) + str(x) +'.png')
                    cv.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)
                    continue
                cropped_img = images[i][y*h:(y+1)*h, x*w:(x+1)*w]
                image_name=(type+str(i) + str(y) + str(x) +'.png')
                print(image_name)
                cv.imwrite(type+ str('_')+str(i).split('.png')[0] + str('_')+ str(y) + str('_') +str(x) +'.png' ,cropped_img)


# method to load images into list
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# custom method to save images to directory folder
def save(img_array, path, name):
  os.chdir(path)
  for i in range(len(img_array)):
    cv2.imwrite(str(name) + str(i) + '.png', img_array[i] )
    
def create_dir(path):
    if os.path.exists(path):
        shutil. rmtree(path)
    os.makedirs(path)    
	
	
########main function start here


for it in os.scandir(current_path):
    if it.is_dir() and it.name[-3:] =='ITE': # this part checks all the directories that contains figures
        os.chdir(os.path.join(current_path, it))
        print(os.getcwd())
        new_path=os.path.join(current_path, it)

        if it.name[0:4]=='mask':
            images=load_images_from_folder(new_path,write=False)
        else:
            images=load_images_from_folder(new_path,type=it.name,write=True)
        
        path=new_path+'/size_changed' 
        if  os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path) 

        save_resized_images_to_folder(images,path)
        saved_path=new_path+'/cropped'

        if  os.path.exists(saved_path):
            shutil.rmtree(saved_path)
        os.mkdir(saved_path) 

 
        name2=(''.join(it.name.split('_')[1:]))
        print('name2------------------',name2)

        if it.name[0:4]=='mask':
            print('file names that will be writte: ',it.name)
            cropSave(images,str(name2), height, weight, saved_path,write=True)
        else:
            cropSave(images,str(name2), height, weight, saved_path,write=False)

        os.chdir(current_path)




bainite_dir = os.path.join(base_dir, 'bainite_set')
martensite_dir = os.path.join(base_dir, 'martensite_set')

# define training and testing directories
train_dir_mask = os.path.join(train_dir, 'mask')
train_dir_original = os.path.join(train_dir, 'original')
validation_dir_mask = os.path.join(validation_dir, 'mask')
validation_dir_original = os.path.join(validation_dir, 'original')
test_dir_mask = os.path.join(test_dir, 'mask')
test_dir_original = os.path.join(test_dir, 'original')

bainite_dir_original = os.path.join(bainite_dir, 'original')
martensite_dir_original = os.path.join(martensite_dir, 'original')

bainite_dir_mask = os.path.join(bainite_dir, 'mask')
martensite_dir_mask = os.path.join(martensite_dir, 'mask')

    
create_dir(train_dir_mask)
create_dir(train_dir_original)
create_dir(validation_dir_mask)
create_dir(validation_dir_original)
create_dir(test_dir_mask)
create_dir(test_dir_original)
create_dir(bainite_dir_original)
create_dir(martensite_dir_original)
create_dir(bainite_dir_mask)
create_dir(martensite_dir_mask)

 



# define cropped images directory

original_N5_440_TMITE_dir = os.path.join(base_dir, 'original_N5_440_TMITE/cropped')
original_N5_325_LBITE_dir = os.path.join(base_dir, 'original_N5_325_LBITE/cropped')
original_42CrMo4_LBITE_dir = os.path.join(base_dir, 'original_42CrMo4_LBITE/cropped')
mask_N5_440_TMITE_dir = os.path.join(base_dir, 'mask_N5_440_TMITE/cropped')
mask_N5_325_LBITE_dir = os.path.join(base_dir, 'mask_N5_325_LBITE/cropped')
mask_42CrMo4_LBITE_dir = os.path.join(base_dir, 'mask_42CrMo4_LBITE/cropped')
 



# load images
original_N5_440_TMITE_dataset = load_images_from_folder(original_N5_440_TMITE_dir)
original_N5_325_LBITE_dataset = load_images_from_folder(original_N5_325_LBITE_dir)
original_42CrMo4_LBITE_dataset=load_images_from_folder(original_42CrMo4_LBITE_dir)
mask_N5_440_TMITE_dataset=load_images_from_folder(mask_N5_440_TMITE_dir)
mask_N5_325_LBITE_dataset=load_images_from_folder(mask_N5_325_LBITE_dir)
mask_42CrMo4_LBITE_dataset=load_images_from_folder(mask_42CrMo4_LBITE_dir)


# check number of images
print(len(original_N5_440_TMITE_dataset))
print(len(original_N5_325_LBITE_dataset))
print(len(original_42CrMo4_LBITE_dataset))
print(len(mask_N5_440_TMITE_dataset))
print(len(mask_N5_325_LBITE_dataset))
print(len(mask_42CrMo4_LBITE_dataset))



# split dataset into training and testing

X_train1, X_rem, y_train1, y_rem  = train_test_split(original_N5_440_TMITE_dataset, mask_N5_440_TMITE_dataset, train_size=0.7, shuffle=True, random_state=69)
X_valid1, X_test1, y_valid1, y_test1= train_test_split(X_rem,y_rem,test_size=0.5, shuffle=True, random_state=69)

X_train2, X_rem, y_train2, y_rem  = train_test_split(original_N5_325_LBITE_dataset, mask_N5_325_LBITE_dataset, train_size=0.7, shuffle=True, random_state=69)
X_valid2, X_test2, y_valid2, y_test2= train_test_split(X_rem,y_rem,test_size=0.5, shuffle=True, random_state=69)

X_train3, X_rem, y_train3, y_rem  = train_test_split(original_42CrMo4_LBITE_dataset,mask_42CrMo4_LBITE_dataset, train_size=0.7, shuffle=True, random_state=69)
X_valid3, X_test3, y_valid3, y_test3= train_test_split(X_rem,y_rem,test_size=0.5, shuffle=True, random_state=69)



# check number of train and test
print(len(X_train1))
print(len(y_train1))
print(len(X_train2))
print(len(y_train2))
print(len(X_train3))
print(len(y_train3))




# save train and test cropped images to their corresponding folders

save(X_train1, train_dir_original, 'N5_440_TMITE_train' )
save(y_train1, train_dir_mask, 'N5_440_TMITE_train' )
save(X_train2,train_dir_original, 'N5_325_LBITE_train' )
save(y_train2,train_dir_mask, 'N5_325_LBITE_train' )
save(X_train3, train_dir_original, '42CrMo4_LBITE_train' )
save(y_train3, train_dir_mask, '42CrMo4_LBITE_train' )



save(X_valid1, validation_dir_original, 'N5_440_TMITE_validation' )
save(y_valid1, validation_dir_mask, 'N5_440_TMITE_validation' )
save(X_valid2,validation_dir_original, 'N5_325_LBITE_validation' )
save(y_valid2,validation_dir_mask, 'N5_325_LBITE_validation' )
save(X_valid3,validation_dir_original, '42CrMo4_LBITE_validation' )
save(y_valid3,validation_dir_mask, '42CrMo4_LBITE_validation' )


save(X_test1, test_dir_original, 'N5_440_TMITE_test' )
save(y_test1, test_dir_mask, 'N5_440_TMITE_test' )
save(X_test2,test_dir_original, 'N5_325_LBITE_test' )
save(y_test2,test_dir_mask, 'N5_325_LBITE_test' )
save(X_test3, test_dir_original, '42CrMo4_LBITE_test' )
save(y_test3, test_dir_mask, '42CrMo4_LBITE_test' )





# Save to Bainite and Martensite

save(X_train1,martensite_dir_original, 'N5_440_TMITE_train' )
save(X_train2,bainite_dir_original, 'N5_325_LBITE_train' )
save(X_valid1,martensite_dir_original, 'N5_440_TMITE_validation' )
save(X_valid2,bainite_dir_original, 'N5_325_LBITE_validation' )
save(X_test1,martensite_dir_original, 'N5_440_TMITE_test' )
save(X_test2,bainite_dir_original, 'N5_325_LBITE_test' )




save(y_train1, martensite_dir_mask, 'N5_440_TMITE_train' )
save(y_train2,bainite_dir_mask, 'N5_325_LBITE_train' )
save(y_valid1,martensite_dir_mask, 'N5_440_TMITE_validation' )
save(y_valid2,bainite_dir_mask, 'N5_325_LBITE_validation' )
save(y_test1,martensite_dir_mask, 'N5_440_TMITE_test' )
save(y_test2,bainite_dir_mask, 'N5_325_LBITE_test' )
