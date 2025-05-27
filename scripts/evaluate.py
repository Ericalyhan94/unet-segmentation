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

def compute_mean_iou(confusion_matrix):
    # """Compute the mean intersection-over-union via the confusion matrix."""
     sum_over_row = (tf.reduce_sum(confusion_matrix, 0))
     sum_over_col = (tf.reduce_sum(confusion_matrix, 1))
     cm_diag = (tf.linalg.tensor_diag_part(confusion_matrix))
     denominator = sum_over_row + sum_over_col - cm_diag
     iou = tf.divide(cm_diag, denominator)
     return iou[1]
     

def write_image(image,iou,filename):
    with open(filename, "a") as my_file:
        my_file.write('\n')  
        my_file.write('####################################################')  
        my_file.write('\n')  
        my_file.write('IoU equals:{}'.format(str(iou)))
        my_file.write('\n')  
    for i in image:
        for j in i:
            for k in j:
                k=str(k)
                with open(filename, "a") as my_file:
                    my_file.write(k)
                    my_file.write('\t')
            with open(filename, "a") as my_file:
                    my_file.write('\n')    

def create_mask(pred_mask):
  #  print('pred_mask11111',pred_mask.shape)
    pred_mask = tf.argmax(pred_mask, axis=-1) #inner most element, index of the maximum value
   # print('pred_mask22222',pred_mask.shape)
    pred_mask = pred_mask[..., tf.newaxis]
   # print('pred_mask333333',np.max(pred_mask[0]))
    return pred_mask[0]


# carbide type number: 13 ,matrix:0

def create_mask2(pred_mask,true_mask):
  #  print('pred_mask11111',pred_mask.shape)
    pred_mask = tf.argmax(pred_mask, axis=-1) #inner most element, index of the maximum value
   # print('pred_mask22222',pred_mask.shape)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask=pred_mask[0]
    #print(pred_mask)
    difference=tf.subtract(pred_mask.numpy(),true_mask[0].numpy())
    #print('difference ',list(set(np.ravel(difference))))
    print('summation False Negative: ',np.sum(difference<0))
    pred_mask=tf.where(difference<0, 25, pred_mask ).numpy()
    #print(mask)
    pred_mask=tf.where(difference>0, 3, pred_mask ).numpy()
    print('summation False Positive: ',np.sum(difference>0))
    #print('mask ',list(set(np.ravel(pred_mask))))
    return pred_mask


def manipulation_mask(mask):
  if not isinstance(mask, np.ndarray):
    mask=mask.numpy()
  output_mask=np.zeros([mask.shape[0],mask.shape[1],3])
  
  for i in np.arange((mask.shape[0])):
    for j in np.arange((mask.shape[1])):
      if mask[i,j]==0:
        output_mask[i,j]=[216,191,216]
      elif mask[i,j]==13:
        output_mask[i,j]=[255,255,0] 
      elif mask[i,j]==3: #False Positive Red
        output_mask[i,j]=[255, 0, 0]
      elif mask[i,j]==25: #False Negative
        output_mask[i,j]=[0, 255, 0]
  return output_mask


def show_predictions(dataset=None, num=1):
    
    if dataset:
        mm=0
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
 
            merged1=cv2.addWeighted(np.array(image[0].numpy().astype(float))*255, 0.7, manipulation_mask(mask[0]).astype(float), 0.4, 0)
            merged2=cv2.addWeighted(np.array(image[0].numpy().astype(float))*255, 0.7, manipulation_mask(create_mask(pred_mask)).astype(float), 0.4, 0)
            merged3=cv2.addWeighted(np.array(image[0].numpy().astype(float))*255, 0.7, manipulation_mask(create_mask2(pred_mask,mask)).astype(float), 0.4, 0)
            display2([image[0],mask[0], create_mask(pred_mask),create_mask2(pred_mask,mask)],mm) #[image[0], mask[0], create_mask(pred_mask),create_mask2(pred_mask,mask),merged1,merged2,merged3]
            #print('image0',merged2)
            mm+=1
            y_true=((mask[0]/13).numpy().astype(int)).reshape(-1)
            y_predict=(create_mask(pred_mask)/13).numpy().astype(int).reshape(-1)
            output = confusion_matrix(y_true, y_predict)
          
            iou_result=compute_mean_iou(output)
            print('iou_result',iou_result)
            
    else:
        display2([sample_image, sample_mask,
             create_mask2(unet.predict(sample_image[tf.newaxis, ...]),mask)],0)

show_predictions(train_dataset, 50)

 # This part is for getting IOU


def show_predictions3(dataset=None, num=1):
    iou_list=[]
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            y_true=((mask[0]/13).numpy().astype(int)).reshape(-1)
            y_predict=(create_mask(pred_mask)/13).numpy().astype(int).reshape(-1)
            output = confusion_matrix(y_true, y_predict)
            iou_result=compute_mean_iou(output)

            merged1=np.array(image[0].numpy().astype(float))*255
            merged2=manipulation_mask(create_mask2(pred_mask,mask)).astype(float)
            merged3=cv2.addWeighted(np.array(image[0].numpy().astype(float))*255, 0.7, manipulation_mask(create_mask2(pred_mask,mask)).astype(float), 0.4, 0)

            if float(iou_result)<0.75:
                write_image(merged1,iou_result,'merged1.txt')
                write_image(merged2,iou_result,'merged2.txt')
                write_image(merged3,iou_result,'merged3.txt')

            iou_list.append(iou_result.numpy())
    return iou_list


   
def main():
    # 1. 加载配置
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    img_size = config["img_size"]
    batch_size = config["batch_size"]
    val_img_dir = base_dir / config["val_image_dir"]
    val_mask_dir = base_dir / config["val_mask_dir"]

    # 2. 加载验证集
    val_ds = get_dataset(val_img_dir, val_mask_dir, img_size, batch_size)

    # 3. 构建模型
    model = build_unet(input_shape=(img_size, img_size, 3))

    # 4. 加载模型权重（修改下面的文件名以匹配你保存的模型）
    checkpoint_path = base_dir / "checkpoints" / "unet_model.h5"
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[dice_coef, iou_score]
    )
    model.load_weights(str(checkpoint_path))

    # 5. 模型评估
   ###################################
    results_train=show_predictions3(processed_image_ds_train.cache().shuffle(BUFFER_SIZE).batch(1), train_image_number)    
    results_validation=show_predictions3(processed_image_ds_validation.cache().shuffle(BUFFER_SIZE).batch(1), validation_image_number)
    results_test=show_predictions3(processed_image_ds_test.cache().shuffle(BUFFER_SIZE).batch(1), test_image_number)
    print('results_train----------------------\n',results_train)
    print('results_validation-------------------\n',results_validation)
    print('results_test-----------------\n',results_test)  
    
    


    with open('Total_IoU.txt', "w") as my_file:
        my_file.write('results_train') 
        my_file.write('\n')
        for i in results_train:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')
        
        my_file.write('results_validation') 
        my_file.write('\n')
        for i in results_validation:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')   

        my_file.write('results_test') 
        my_file.write('\n')
        for i in results_test:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')

    
    fig, ax = plt.subplots(figsize=(12,8))
 
    plt.rcParams["font.serif"] = ["Times New Roman"]
    csfont = {'fontname':'Times New Roman'}
    plt.scatter(np.arange(train_image_number),results_train)
    plt.scatter(train_image_number+np.arange(validation_image_number),results_validation)
    plt.scatter(validation_image_number+train_image_number+np.arange(test_image_number),results_test)
    plt.legend(['Training Data','Validation Data','Testing Data'],fontsize=24,loc=[1.02,0.5])
    plt.xlabel('Image Number',fontsize=20,**csfont)
    plt.ylabel('IoU Value - Carbides',fontsize=20,**csfont)
    plt.legend(['Training Data','Validation Data','Testing Data'],loc='upper center', bbox_to_anchor=(0.5, 1.1),fancybox=True, shadow=True, ncol=5,fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10,direction='in', )
    #ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)

    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.savefig('IoUPlot.png')
    plt.show()


    fig, [ax1,ax2] = plt.subplots(1,2)
    fig.set_figwidth(18)
    fig.set_figheight(9)
 

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.rcParams["font.serif"] = ["Times New Roman"]
    csfont = {'fontname':'Times New Roman'}
    ax1.plot(acc, label='Training Accuracy',linewidth=2)
    ax1.plot(val_acc, label='Validation Accuracy',linewidth=2)
    ax2.plot(loss, label='Training Loss',linewidth=2)
    ax2.plot(val_loss, label='Validation Loss',linewidth=2)
 

    with open('LossAndAccuracy.txt', "w") as my_file:
        my_file.write('acc') 
        my_file.write('\n')
        for i in acc:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')
        
        my_file.write('val_acc') 
        my_file.write('\n')
        for i in val_acc:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')   
        my_file.write('loss') 
        my_file.write('\n')
        for i in loss:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')
        my_file.write('val_loss') 
        my_file.write('\n')
        for i in val_loss:
            i=str(i)
            my_file.write(i)  
            my_file.write(',')
        my_file.write('\n')
    
    


    ax1.set_xlabel('Epoch Number',fontsize=24,**csfont)
    ax1.set_ylabel('Accuracy',fontsize=24,**csfont)
    ax2.set_xlabel('Epoch Number',fontsize=24,**csfont)
    ax2.set_ylabel('Loss',fontsize=24,**csfont)



 
    ax1.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10,direction='in', )
    ax2.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10,direction='in', )

    a=1.12
    aa=-0.02
    ax1.legend(['Training Data','Validation Data'],loc='upper left', bbox_to_anchor=(aa, a),fancybox=True, shadow=True, ncol=5,fontsize=20)
    ax2.legend(['Training Data','Validation Data'],loc='upper left', bbox_to_anchor=(aa, a),fancybox=True, shadow=True, ncol=5,fontsize=20)


    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    [x.set_linewidth(1.5) for x in ax2.spines.values()]


    plt.savefig('LossAndAccuracy.png')
    plt.show()
 

if __name__ == "__main__":
    main()