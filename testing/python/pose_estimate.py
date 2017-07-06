import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
import caffe
from config_reader import config_reader
import pose_detect
import time
import numpy as np
import pandas as pd


param, model = config_reader()

if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

#run the test for images
#home_dir = '/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/images'
home_dir = '/data0/krohitm/code/temp_imgs'
_,folders,_ = os.walk(home_dir).next()
folders.sort()

columns_angles = ['right_collar_angle', 'left_collar_angle', 'right_upper_arm_angle',
               'right_forearm_angle', 'left_upper_arm_angle', 'left_forearm_angle', 
               'neck_to_right_hip_angle', 'right_thigh_angle', 'right_calf_angle', 
               'neck_to_left_hip_angle', 'left_thigh_angle', 'left_calf_angle',
               'neck_angle', 'right_eye_angle', 'left_eye_angle', 
               'right_eye_to_back_angle', 'left_eye_to_back_angle']
columns_lengths = ['right_collar_length', 'left_collar_length', 'right_upper_arm_length',
               'right_forearm_length', 'left_upper_arm_length', 'left_forearm_length', 
               'neck_to_right_hip_length', 'right_thigh_length', 'right_calf_length', 
               'neck_to_left_hip_length', 'left_thigh_length', 'left_calf_length', 
               'neck_length', 'right_eye_length', 'left_eye_length', 
               'right_eye_to_back_length', 'left_eye_to_back_length']
columns_parts = ['Neck','skip1','RShoulder','RElbow','LShoulder','LElbow', 
                     'skip2','RHip', 'RKnee','skip3', 'LHip', 'LKnee', 'skip4', 
                     'Nose', 'REye', 'LEye']

#bbox = []
angles = pd.DataFrame(columns=columns_angles)
lengths = pd.DataFrame(columns=columns_lengths)
body_parts= pd.DataFrame(columns=columns_parts)
#array to store img_info as [image_name, dict(angles_of_limbs), dict(lengths_of_limbs)]
img_info = []
features = pd.DataFrame()
img_names = pd.DataFrame(columns=['image_name'])

img_index = 0
for folder in folders:
    #try:
    #    os.mkdir('/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/pose_detections_PAF/'+folder)
    #except OSError:
    #    print "Directory {0} already exists".format(folder)

    _,_,imgs = os.walk(os.path.join(home_dir,folder)).next()
    imgs.sort()
    #count = 0
    flag = 0
    for img in imgs:
        start_time = time.time()
        full_image_path = os.path.join(os.path.join(home_dir, folder), img)
        #if full_image_path == '/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/images/2017-05-19-1143-43/0013414.jpg':
        #    flag = 1
        #if flag == 0:
        #    continue
        _, angles_temp, lengths_temp, body_parts_temp = pose_detect.pose_detect(param, net, model, full_image_path)
        print "detection done for {}".format(full_image_path)
        print "total time for this image was %.4f s." %((time.time() - start_time))
        print "*************************************************************************"
        #check if no bbox detected
        #if bbox_temp == []:
        #    bbox.append([-1,-1,-1,-1])
        #    continue
        #bbox.append(bbox_temp)
        #angles.append(angles_temp)
        #lengths.append(lengths_temp)
        img_names.set_value(img_index, 'image_name', full_image_path)
        #img_names = img_names.append(full_image_path, ignore_index=True)
        angles = angles.append(angles_temp, ignore_index=True)
        lengths = lengths.append(lengths_temp, ignore_index=True)
        body_parts = body_parts.append(body_parts_temp, ignore_index=True)
        #print angles
        #print angles_temp
        #img_info.append([full_image_path, angles_temp, lengths_temp])
        #count += 1
        img_index += 1
        #if count >=3:
        #    break
        

all_features = [img_names, body_parts, angles, lengths]
combs = pd.concat(all_features, axis = 1)
#print combs.loc[0,'image_name']
#print angles
#print lengths
#print img_info
combs.drop(['skip1','skip2','skip3','skip4'], axis=1, inplace=True)
combs.to_csv('check.csv', index=False)
