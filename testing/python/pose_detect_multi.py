import cv2 as cv 
import numpy as np
import math
import util
import matplotlib
import pylab as plt
from scipy.ndimage.filters import gaussian_filter
import pandas as pd

def pose_detect(param, net, model, full_img_name, storage_loc):
    #start_time = time.time()
    test_image = full_img_name
    oriImg = cv.imread(test_image) # B,G,R order
    #print oriImg.shape
    
    #resizing to 368*654 as mentioned in the paper
    resize_x = 0.340625
    resize_y = 0.340740
    
    oriImg = cv.resize(oriImg, None, fx = resize_x, fy = resize_y, 
                       interpolation = cv.INTER_CUBIC)
    
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    imageToTest = oriImg
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
    net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
    net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
    #print "Image processing took %.2f ms. " %(1000 * (time.time() - start_time))
    
    #run the model to get heat maps and pafs
    #start_time = time.time()
    output_blobs = net.forward()
    #print ('The CNN took %.2f ms. ' %(1000 * (time.time() - start_time)))
    
    
    # extract outputs, resize, and remove padding
    #start_time = time.time()
    heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
    heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
    paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    #print ('Extracting output, resizing and removing padding took %.2f ms. ' %(
    #        1000 * (time.time() - start_time)))
    
    
    heatmap_avg = heatmap
    paf_avg = paf
    all_peaks = []
    peak_counter = 0
    
    for part in range(19-1):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=2)
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
       
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, 
                                              map>=map_up, map>=map_down, 
                                              map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        #print id
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in 
                                   range(len(id))]
        
        #print peaks_with_score_and_id
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
        #print peak_counter
    #print "Finding peaks for joints took %.2f ms. " %(1000 * (time.time() - start_time))


    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], [10,11],
               [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], 
               [16,18], [3,17], [6,18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], 
              [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], 
              [53,54], [51,52], [55,56], [37,38], [45,46]]

    connection_all = []
    special_k = []
    mid_num = 10
    
    
    #start_time = time.time()
    #store paf candidates
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    
                    #add for zero division error
                    norm = norm + 1e-5
                    vec = np.divide(vec, norm)
                    
                    startend = zip(np.linspace(candA[i][0], candB[j][0], 
                                               num=mid_num), np.linspace(
                                                       candA[i][1], candB[j][1],
                                                       num=mid_num))
                    
                    vec_x = np.array([score_mid[int(round(startend[I][1])), 
                                                int(round(startend[I][0])), 0
                                                ] for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), 
                                                int(round(startend[I][0])), 1
                                                ] for I in range(len(startend))])
                    
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts
                                               ) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, 
                                                     score_with_dist_prior+
                                                     candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    
    #print "PAF took %.2f ms. " %(1000 * (time.time() - start_time))


    #start_time = time.time()
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
            
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    #print "PAF2 took %.2f ms. " %(1000 * (time.time() - start_time))
    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    """PUT HERE ONLY FOR USING WITH COUNTINGPEOPLE.PY"""
    plt.close()
    return len(subset)
    # visualize
    #start_time = time.time()
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
              [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
              [255, 0, 170], [255, 0, 85]]

    """all_peaks is in the form [joint][person][coordinates, confidence, part number]"""
    

    img_name_split = full_img_name.split('/')
    folder_name = img_name_split[-2]
    img_name = img_name_split[-1]
    
    #draw the sticks for the limbs
    canvas = cv.imread(test_image) # B,G,R order
    canvas = cv.resize(canvas, None, fx = resize_x, fy = resize_x, 
                       interpolation = cv.INTER_CUBIC)
    
    
    stickwidth = 4
    
    """order of limbs is: right collar, left collar, right upper arm,
    right forearm, left upper arm, left forearm, neck to right hip, 
    right thigh, right calf, neck to left hip, left thigh, left calf, neck, right eye,
    left eye, right eye to back, left eye to back, """
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
    
    #n = max_area_index
    #angles = {}
    #lengths = {}
    body_parts = pd.DataFrame(columns=columns_parts)
    angles = pd.DataFrame(columns=columns_angles)
    lengths = pd.DataFrame(columns=columns_lengths)
    for n in range (len(subset)):
        #x_min = 10000
        #y_min = 10000
        #x_max = -1
        #y_max = -1
        for i in range(17):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            X = candidate[index.astype(int), 0]
            Y = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv.ellipse2Poly((int(mX),int(mY)), (int(length/2), 
                                       stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            cv.circle(cur_canvas, (int(X[0]), int(Y[0])), 4, colors[i], 
                      thickness=-1)
            cv.circle(cur_canvas, (int(X[1]), int(Y[1])), 4, colors[i], 
                      thickness=-1)
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            angles.set_value(n, columns_angles[i], angle)
            lengths.set_value(n, columns_lengths[i], length)
            if i < 16:
                body_parts.set_value(n, columns_parts[i], (X[0], Y[0]))
            
        #    if X[0] < x_min or X[1] < x_min:
        #        x_min = min(X)
        #    if Y[0] < y_min or Y[1] < y_min:
        #        y_min = min(Y)
        #    if X[1] > x_max or X[0] > x_max:
        #        x_max = max(X)
        #    if Y[1] > y_max or Y[0] > y_max:
        #        y_max = max(Y)
            
        #img_bounds = canvas.shape
        #x_min = max(int(x_min - 18), 0)
        #y_min = max(int(y_min - 18), 0)
        #x_max = min(int(x_max + 18), img_bounds[1])
        #y_max = min(int(y_max + 18), img_bounds[0])
        #cv.rectangle(canvas, (x_min, y_min), (x_max, y_max), colors[n], thickness=2)
    
    
    #plt.imshow(canvas[:,:,[2,1,0]])
    #plt.axis('off')
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(12, 12)
    #fig.savefig(storage_loc+'/'+folder_name+'/'+img_name)
    plt.close()
    return angles, lengths, body_parts