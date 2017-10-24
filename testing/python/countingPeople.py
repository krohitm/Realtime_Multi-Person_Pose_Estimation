#import sys 
import os
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import caffe
from config_reader import config_reader
import pose_detect_multi
import time
#import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Folder locations')
parser.add_argument('--images_location', dest='images_location', 
                    help='home directory of images', type=str)
#parser.add_argument('--store_location', dest='store_location', 
#                    help='home directory to store images', type=str)

args = parser.parse_args()


param, model = config_reader()

if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

#run the test for images
#home_dir = '/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/images'
home_dir = args.images_location#'/data0/krohitm/code/temp_imgs'
#storage_loc = args.store_location
_,folders,_ = os.walk(home_dir).next()
folders.sort()

combs = pd.DataFrame()

day = 1
for folder in folders:
    #try:
    #    os.mkdir(os.path.join(storage_loc, folder))#'/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/pose_detections_PAF/'+folder)
    #except OSError:
    #    print "Directory {0} already exists".format(folder)

    _,_,imgs = os.walk(os.path.join(home_dir,folder)).next()
    imgs.sort()
    if "peopleCount.csv" in imgs:
        continue
    
    for img in imgs:
        img_names = pd.DataFrame(columns=['image_name'])
        start_time = time.time()
        full_image_path = os.path.join(os.path.join(home_dir, folder), img)
        noOfPeople = pd.DataFrame(columns=['number_of_people'])
        noOfPeople.set_value(0, 'number_of_people', pose_detect_multi.pose_detect(
                param, net, model, full_image_path, home_dir))
        print "detection done for {}".format(full_image_path)
        print "total time for this image was %.4f s." %((time.time() - start_time))
        print "*************************************************************************"
        
        img_names.set_value(0, 'image_name', img)
        comb_temp = pd.concat([img_names, noOfPeople],
                              axis = 1)
        combs = combs.append(comb_temp, ignore_index=True)
    
    combs.to_csv(os.path.join(home_dir, folder, 'peopleCount.csv'), index=False)