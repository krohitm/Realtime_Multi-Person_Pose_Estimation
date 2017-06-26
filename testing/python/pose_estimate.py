import sys, os
sys.path.append('/usr/local/lib/python2.7/site-packages')
import caffe
from config_reader import config_reader
import pose_detect
import time


param, model = config_reader()

if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

#run the test for images
home_dir = '/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/images'
_,folders,_ = os.walk(home_dir).next()
folders.sort()
for folder in folders:
    try:
        os.mkdir('/home/krohitm/code/Realtime_Multi-Person_Pose_Estimation/testing/pose_detections_PAF/'+folder)
    except OSError:
        print "Directory {0} already exists".format(folder)

    _,_,imgs = os.walk(os.path.join(home_dir,folder)).next()
    imgs.sort()
    for img in imgs:
        start_time = time.time()
        full_image_path = os.path.join(os.path.join(home_dir, folder), img)
        pose_detect.pose_detect(param, net, model, full_image_path)
        print "detection done for {}".format(full_image_path)
        print "total time for this image was %.4f s." %((time.time() - start_time))
        print "*************************************************************************"
