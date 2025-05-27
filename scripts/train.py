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
from models.unet import build_unet

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
   # img=imageio.imread(image_path)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    print(mask)
    mask = tf.image.decode_png(mask, channels=3)
    #mask = mask_convertion(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    #mask = mask_convertion(mask)
    #mask=imageio.imread(mask_path)
  
    mask = int(mask[:,:,1]>0.3)*13
    
    mask=tf.reshape(mask, [480,640,1])
    
 #   mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')


    return input_image, input_mask

def main():
    # 1. 加载配置文件
    base_directory = Path(__file__).resolve().parent.parent
    config_path = base_directory / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. 提取配置参数
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    
    img_height=config["img_height"]
    img_width=config["img_width"]
    num_channels=config["num_channels"]
    
    image_path_train = base_dir / config["train_image_dir"]
    mask_path_train = base_dir / config["train_mask_dir"]
    image_path_validation = base_dir / config["val_image_dir"]
    mask_path_validation = base_dir / config["val_mask_dir"]
    image_path_test= base_dir / config["test_image_dir"]
    mask_path_test=base_dir / config["test_mask_dir"]
    
    bainite_image_path_train =base_dir / config["bainite_image_dir"]
    bainite_mask_path_train=base_dir / config["bainite_mask_dir"]
    
    martensite_image_path_train=base_dir / config["martensite_image_dir"]
    martensite_mask_path_train = base_dir / config["martensite_mask_dir"]

    lower_bainite_for_size=base_dir/config["lower_bainite_for_size"]
    tempered_martensite_for_size=base_dir/config["tempered_martensite_for_size"]
    
    EPOCHS=config["EPOCHS"]
    VAL_SUBSPLITS=config["VAL_SUBSPLITS"]
    BUFFER_SIZE=config["BUFFER_SIZE"]
    BATCH_SIZE=config["BATCH_SIZE"]

    
    
    # 3. construct dataset
    
    image_list_train = os.listdir(image_path_train)
    image_list_validation = os.listdir(image_path_validation)
    image_list_test = os.listdir(image_path_test)
    
    
    mask_list_train = os.listdir(mask_path_train)
    mask_list_validation = os.listdir(mask_path_validation)
    mask_list_test = os.listdir(mask_path_test)
    
    
    image_list_train = [image_path_train+i for i in image_list_train]
    mask_list_train = [mask_path_train+i for i in mask_list_train]
 
    image_list_validation = [image_path_validation+i for i in image_list_validation]
    mask_list_validation = [mask_path_validation+i for i in mask_list_validation]

    image_list_test = [image_path_test+i for i in image_list_test]
    mask_list_test = [mask_path_test+i for i in mask_list_test]
    
    image_list_train.sort()
    image_list_validation.sort()
    image_list_test.sort()


    mask_list_train.sort()
    mask_list_validation.sort()
    mask_list_test.sort()

    train_image_number=len(image_list_train)
    validation_image_number=len(image_list_validation)
    test_image_number=len(image_list_test)

    image_list_ds_train = tf.data.Dataset.list_files(image_list_train, shuffle=False)
    mask_list_ds_train = tf.data.Dataset.list_files(mask_list_train, shuffle=False)

    image_list_ds_validation = tf.data.Dataset.list_files(image_list_validation, shuffle=False)
    mask_list_ds_validation = tf.data.Dataset.list_files(mask_list_validation, shuffle=False)

    image_list_ds_test = tf.data.Dataset.list_files(image_list_test, shuffle=False)
    mask_list_ds_test = tf.data.Dataset.list_files(mask_list_test, shuffle=False)
    
    
    

    for path in zip(image_list_ds_train.take(3), mask_list_ds_train.take(3)):
        print(path)

    image_filenames_train = tf.constant(image_list_train)
    masks_filenames_train = tf.constant(mask_list_train)
    image_filenames_validation = tf.constant(image_list_validation)
    masks_filenames_validation = tf.constant(mask_list_validation)
    image_filenames_test = tf.constant(image_list_test)
    masks_filenames_test = tf.constant(mask_list_test)



    dataset_train = tf.data.Dataset.from_tensor_slices((image_filenames_train, masks_filenames_train))
    dataset_validation = tf.data.Dataset.from_tensor_slices((image_filenames_validation , masks_filenames_validation))
    dataset_test = tf.data.Dataset.from_tensor_slices((image_filenames_test, masks_filenames_test))


    image_ds_train = dataset_train.map(process_path)
    image_ds_validation = dataset_validation.map(process_path)
    image_ds_test = dataset_test.map(process_path)


    processed_image_ds_train = image_ds_train.map(preprocess)
    processed_image_ds_validation = image_ds_validation.map(preprocess)
    processed_image_ds_test = image_ds_test.map(preprocess)
    
    
    unet = unet_model((img_height, img_width, num_channels))
    # 4 .construct the model 
    unet.summary()
    
    unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 
    
    
    
    ########################################################
    
  

    # 5. 设置回调函数
    
    
    processed_image_ds_train.batch(BATCH_SIZE)
    train_dataset = processed_image_ds_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    validation_dataset= processed_image_ds_validation.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    print(processed_image_ds_train.element_spec)


    # load existing model if exist


    checkpoint_path = base_dir / "checkpoints" / "unet_model.ckpt"
    if os.path.exists(checkpoint_path+'.index'):
        print('---------------------------load the model-----------------------------')
        unet.load_weights(checkpoint_path)
	

    fine_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True)


    model_history = unet.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset,callbacks=[fine_callback])

if __name__ == "__main__":
    main()