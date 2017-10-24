import pandas as pd
import cv2 as cv
#from ast import literal_eval as make_tuple
import os, argparse, sys

parser = argparse.ArgumentParser(description='Folder location')
parser.add_argument('--store_location', dest='store_location', 
                    help='home directory to store images', type=str)

args = parser.parse_args()
storage_loc = args.store_location

def write_to_csv(folder):
    allCounts = pd.DataFrame(peopleCount, columns=['peopleCount'])
    allCounts_info = pd.concat([data, allCounts], axis=1)
    allCounts_info.to_csv(os.path.join(storage_loc, folder, 'peopleCount.csv'), index=False)
    sys.exit()

_,folders,_ = os.walk(storage_loc).next()
folders.sort()

#resizing to 368*654 as mentioned in the paper
resize_x = 0.340625
resize_y = 0.340740
    
for folder in folders:
    _,_,images = os.walk(os.path.join(storage_loc,folder)).next()
    images.sort()
    if os.path.exists(os.path.join(storage_loc, folder, 'peopleCount.csv')):
        data = pd.read_csv(os.path.join(storage_loc, folder, 'peopleCount.csv'),
                                    delimiter=',', dtype={'image_name':str})
        peopleCount = list(data['peopleCount'].dropna())
        start = len(peopleCount)
        data.drop(['peopleCount'], inplace=True, axis=1)
    else:
        peopleCount = []
        data = pd.DataFrame(images, columns = ['image_name'])
        start = 0
        #data = pd.DataFrame(columns = ['image_name', 'peopleCount'])
    
    changeCountFlag = 0
    i = start
    while i < len(images):
        img_name = images[i]
        full_img_name = os.path.join(storage_loc, folder, img_name)
    
        canvas = cv.imread(full_img_name) # B,G,R order
        canvas = cv.resize(canvas, None, fx = resize_x, fy = resize_x, 
                           interpolation = cv.INTER_CUBIC)
    
        cv.namedWindow(full_img_name, cv.WINDOW_NORMAL)
        cv.imshow(full_img_name,canvas)
    
        key_flag=0
        print "press number key equal to number of people in image"
        print "press n to skip, w for incorrect points, u for unclear, f for correcting previous pose"
        while key_flag==0:
            key_count = cv.waitKey(30)
            if key_count == ord('f'):
                changeCountFlag = 1
                i -= 1
                key_flag = 1
            elif key_count == ord('s'):
                write_to_csv(folder)
            elif key_count in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
                #print chr(key_count)
                curPeopleCount = chr(key_count)
                key_flag = 1
            
                
        cv.destroyAllWindows()
        if changeCountFlag == 0:
            peopleCount.append(curPeopleCount)
        else:
            peopleCount.pop()
            changeCountFlag = 0
            i -= 1
        
        i += 1
    
    write_to_csv(folder)