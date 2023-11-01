# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 22:30:12 2023

@author: Skywalker_chan
"""

import cv2 as cv, numpy as np
from matplotlib import pyplot as pp
import re # for regex - extract file name from path
import glob # globbing utility - for reading multi img
import sys, os, shutil

def sky_region_detection(path, output_folder, mask_path):
    temp_filename = '' # storing file name (without path but with file type ext)
    temp_filename_only = ''
    day_night_info = []
    processing_count = 0
    mask = cv.imread(mask_path, 0) # reading the provided mask of the dataset (supposedly the expected ground truth)
    overall_accuracy = [] # for calc overall accuracy of a mask
    highest_acc_mask = [] # store the highest accuracy generated mask from the program as it runs
    highest_acc_skyline = []
    highest_acc_value = 0
    highest_acc_filename = ""
    
    # defining all func
    def rgb_info(r_layer,g_layer,b_layer):
        # compare mean for all layers
        mean_r = np.mean(r_layer)
        mean_g = np.mean(g_layer)
        mean_b = np.mean(b_layer)
        # overall mean
        avg_mean = (mean_r + mean_g + mean_b)/3
        [nrow,ncol] = b_layer.shape
        # mean for top 25% region of the image
        mean_25 = np.mean(b_layer[:int(nrow*0.26),:])
        mean_25_r = np.mean(r_layer[:int(nrow*0.26),:])
        mean_25_g = np.mean(g_layer[:int(nrow*0.26),:])
        # overall mean of top 25%
        sky_avg_mean = (mean_25 + mean_25_r + mean_25_g)/3
        return mean_r,mean_g,mean_b,avg_mean,mean_25,sky_avg_mean
    
    def grayscale_info(r_layer):
        # take 1 layer as param enuf (if grayscale -> all layers hv exact same values)
        avg_mean = np.mean(r_layer)
        [nrow,ncol] = r_layer.shape
        sky_avg_mean = np.mean(r_layer[:int(nrow*0.26),:])
        return avg_mean,sky_avg_mean
    
    def day_or_night(day_night_info, img_type):
        if sky_avg_mean > avg_mean and avg_mean >110: # day time img
            day_night_info.append({'file name': temp_filename, 'time of day': 'daytime', 'img type': img_type})
        else:
            day_night_info.append({'file name': temp_filename, 'time of day': 'nighttime', 'img type': img_type})

    # go through each image in the dataset folder
    for file in glob.glob(path):
        image_read = cv.imread(file)
        # obtain avg intensity value from each layer (default -> BGR)
        r_layer = image_read[:,:,2]
        g_layer = image_read[:,:,1]
        b_layer = image_read[:,:,0]
        # get shape
        [nrow, ncol] = image_read.shape[:2]
    
        # check if it is grayscale (c if exactly same or not)
        is_grayscale = True
        for r in range(0, nrow):
            for c in range(0, ncol):
                if (r_layer[r,c] != g_layer[r,c]) or (g_layer[r,c] != b_layer[r,c]):
                    is_grayscale = False
                    break
                
        # get the file name only (remove the path)
        og_filename = file
        raw_string = r"{}".format(og_filename)
        result = re.search(r"\\(.*)" , raw_string)
        # if want to remove the ".jpg" also
        result2 = re.search(r"\\(.*?)\." , raw_string)
        if result:
            temp_filename = result.group(1)
        if result2:
            temp_filename_only = result2.group(1)
            
        print("Current file: ", temp_filename)
        
        # Day night identifier algo
        if is_grayscale:
            [avg_mean,sky_avg_mean] = grayscale_info(r_layer)
            # store info if img is taken during day or night time
            day_or_night(day_night_info, "grayscale")
        else: # not grayscale
            [mean_r,mean_g,mean_b,avg_mean,mean_25,sky_avg_mean] = rgb_info(r_layer, g_layer, b_layer)
            day_or_night(day_night_info, "rgb")
        # ---------------------------------------------------------------------------
        # sky detection
        
        # detecting sky line boundary
        def sky_line_border_point(nrow,ncol,mean_25, avg_mean, b_layer):
            output_img = np.zeros((nrow,ncol), dtype=np.uint8)
            filled_img = np.zeros((nrow,ncol), dtype=np.uint8)
            labelled_img = np.zeros((nrow,ncol), dtype=np.uint8)
            intensity_diff_thres = mean_25-avg_mean
            border_tracker = True
            not_sky_count = 0
            # scan through col by col 
            for c in range(0,ncol):
                for r in range(0,nrow):
                    # if current pixel value difference too large from mean sky blue intensity
                    if abs(mean_25 - b_layer[r,c]) > intensity_diff_thres:
                        not_sky_count+=1
                        if border_tracker:
                            output_img[r,c] = 255
                            border_tracker = False
                    else: # filling logic
                        filled_img[r,c] = 255
                    # when just found sky pixel again
                    if abs(mean_25 - b_layer[r,c]) < intensity_diff_thres and not_sky_count>0:
                        not_sky_count = 0 # reset to find the next non sky area
                        border_tracker = True
                        output_img[r-1,c] = 255
                    # still considered in sky region -> but not detected to be a sky pixel (protruding buildings/objects)
                    if abs(mean_25 - b_layer[r,c]) > intensity_diff_thres and r < nrow*0.4:
                        labelled_img[r,c] = 100
            return output_img, filled_img, labelled_img
        
        def hit_or_miss(input_img):
            sE1 = np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,-1]],dtype=int)
            sE2 = np.array([[-1,-1,-1],[-1,1,1],[-1,1,-1]],dtype=int)
            sE3 = np.array([[-1,-1,-1],[-1,1,-1],[-1,1,-1]],dtype=int)
            fil_1 = cv.morphologyEx(input_img, cv.MORPH_HITMISS, sE1)
            fil_2 = cv.morphologyEx(input_img, cv.MORPH_HITMISS, sE2)
            fil_3 = cv.morphologyEx(input_img, cv.MORPH_HITMISS, sE3)
            return [fil_1, fil_2, fil_3]
        
        # ground truth/mask filling
        def ground_truth_filling(labelled_img, ground_truth):
            for c in range(0,ncol):
                for r in range(0,nrow):
                    if labelled_img[r,c] == 100 and ground_truth[r,c] !=255:
                        ground_truth[r,c] = 255
                    if ground_truth[r,c] == 255 and r > nrow*0.37:
                        if r != nrow-1:
                            if np.all(ground_truth[r:r+6,c-6:c-2]) == 0:
                                continue
                            else:
                                ground_truth[r:,c] = 255
                        continue
                    
        def hit_or_miss_output(input_img, components):
            output_img = input_img.copy()
            for se in components:
                output_img = cv.bitwise_xor(output_img, se)
            return output_img
        
        # build the skyline (more refined & complete) from ground truth/mask produced
        def finalised_skyline(input_img):
            def kernel_sliding(kernel, skyline, r,c):
                if kernel[1,1] == 0:
                    for kr in range(0,3):
                        for kc in range(0,3):
                            if kernel[kr,kc] == 255 and [kr,kc] != [1,1]:
                                skyline[r,c] = 255
                                break
            
            skyline = np.zeros((nrow,ncol), dtype=np.uint8)
            kernel = np.zeros((3,3), dtype=np.uint8)
            for r in range(0,nrow):
                for c in range(0,ncol):
                    # special cases (near the boundary)
                    # if left top corner (r-1, c-1)
                    if r == 0 and c == 0:
                        kernel[1:,1:] = input_img[r:r+2,c:c+2]
                        kernel_sliding(kernel, skyline, r,c)
                    # if right top corner (r-1, c+1)
                    elif r == 0 and c == ncol-1:
                        kernel[1:,:2] = input_img[r:r+2,c-1:c+1]
                        kernel_sliding(kernel, skyline, r,c)
                    # if left bottom corner (r+1, c-1)
                    elif r == nrow-1 and c == 0:
                        kernel[:2,1:] = input_img[r-1:r+1,c:c+2]
                        kernel_sliding(kernel, skyline, r,c)
                    # if right bottom corner (r+1, c+1)
                    elif r == nrow-1 and c == ncol-1:
                        kernel[:2,:2] = input_img[r-1:r+1,c-1:c+1]
                        kernel_sliding(kernel, skyline, r,c)      
                    # if left
                    elif c == 0:
                        kernel[:,1:] = input_img[r-1:r+2,c:c+2]
                        kernel_sliding(kernel, skyline, r,c)
                    # if right
                    elif c == ncol-1:
                        kernel[:,:2] = input_img[r-1:r+2,c-1:c+1]
                        kernel_sliding(kernel, skyline, r,c)
                    # if top
                    elif r == 0:
                        kernel[1:,:] = input_img[r:r+2,c-1:c+2]
                        kernel_sliding(kernel, skyline, r,c)
                    # if bottom
                    elif r == nrow-1:
                        kernel[:2,:] = input_img[r-1:r+1,c-1:c+2]
                        kernel_sliding(kernel, skyline, r,c)
                    # pixel is in center of img
                    else:                   
                        kernel = input_img[r-1:r+2,c-1:c+2]
                        kernel_sliding(kernel, skyline, r,c)
            return skyline
        
        # evaluate the accuracy of the algorithm on the image with the provided mask from dataset
        def accuracy_evaluation(mask, final_product):
            correct_counts = 0
            for r in range(0,nrow):
                for c in range(0,ncol):
                    if final_product[r,c] == mask[r,c]:
                        correct_counts+=1
            # accuracy_result = 
            return (correct_counts/(nrow*ncol))*100
        
        # combine the operations into 1 function
        def combined_operations(sky_boundary_edge, sky_filled, labelled_img, temp_filename, temp_filename_only, highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename):
            structuring_element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            
            # perform closing
            ground_mask = cv.bitwise_not(sky_filled)
            ground_truth = cv.morphologyEx(ground_mask, cv.MORPH_CLOSE, structuring_element, iterations=1)
            # hit_or_miss() function
            ground_truth_components = hit_or_miss(ground_truth)
            # ground_truth_filling() function
            ground_truth_filling(labelled_img, ground_truth)
            # hit_or_miss_output() function
            ground_truth_filtered = hit_or_miss_output(ground_truth, ground_truth_components)
            # make the sky region be white after filling the ground truth/mask
            ground_truth2 = cv.bitwise_not(ground_truth_filtered)
    
            # build the skyline from ground truth/mask produced
            skyline = finalised_skyline(ground_truth2)
    
            pp.figure(temp_filename)
            pp.subplot(1,2,1)
            pp.title("Ground truth")
            pp.imshow(ground_truth2, cmap='gray')
            pp.subplot(1,2,2)
            pp.title("Skyline")
            pp.imshow(skyline, cmap='gray')
            saved_img_name = output_folder + "/" + temp_filename_only + ".png"
            pp.savefig(saved_img_name)
            
            # evaluate the accuracy of the algorithm on the image with the provided mask from dataset
            ground_truth2_acc = accuracy_evaluation(mask, ground_truth2)
            overall_accuracy.append(ground_truth2_acc)
            print("Accuracy reading (Ground truth/mask): ", ground_truth2_acc, "\n\n")
            
            # getting the mask with the highest accuracy (store a copy)
            
            if highest_acc_value == 0: # if no mask is retrieved yet
                highest_acc_mask = ground_truth2.copy()
                highest_acc_value = ground_truth2_acc
                highest_acc_skyline = skyline.copy()
                highest_acc_filename = temp_filename
                
            if ground_truth2_acc > highest_acc_value:
                highest_acc_mask = ground_truth2.copy()
                highest_acc_value = ground_truth2_acc
                highest_acc_skyline = skyline.copy()
                highest_acc_filename = temp_filename
            
            # if len(overall_accuracy) >= 2:
            #     if ground_truth2_acc > overall_accuracy[-2]:
            #         highest_acc_mask = ground_truth2.copy()
            #     else:
            #         highest_acc_mask = ground_truth2.copy()
            # else: # if no mask is retrieved yet
            #     highest_acc_mask = ground_truth2.copy()
            #     # for acc in range(0,len(overall_accuracy)):
            #     #     if ground_truth2_acc < acc:
            #     #         highest_acc_value = acc
            
            
            
            # save the resulting outputs to a folder
            saved_img_name1 = output_folder + "/" + temp_filename_only
            # os.mkdir(saved_img_name1)
            if not os.path.exists(saved_img_name1): # if not created
                os.mkdir(saved_img_name1)
            else: # if ald created
                shutil.rmtree(saved_img_name1)
                os.mkdir(saved_img_name1)
            saved_img_name2 = saved_img_name1 + "/Ground truth.png"
            saved_img_name3 = saved_img_name1 + "/Skyline.png"
            cv.imwrite(saved_img_name2, ground_truth2)
            cv.imwrite(saved_img_name3, skyline)
            
            return highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename
            
        '''
        images to segregate and process:
            - rgb
            - grayscale
            - if night time img -> ignore (skip sky region detection processing)
        '''
        

        # if current image is rbg and day time
        if day_night_info[-1]['file name'] == temp_filename and day_night_info[-1]['img type'] == "rgb" and day_night_info[-1]['time of day'] == "daytime":
            processing_count+=1
            print("Processing queue ", processing_count)
            print("Currently processing image:", temp_filename, "\n\n")
            # sky border point comparison
            sky_boundary_edge, sky_filled, labelled_img = sky_line_border_point(nrow, ncol, mean_25, avg_mean, b_layer)
            
            highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename = combined_operations(sky_boundary_edge, sky_filled, labelled_img, temp_filename, temp_filename_only, highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename)
        
        # if current image is grayscale and day time
        elif day_night_info[-1]['file name'] == temp_filename and day_night_info[-1]['img type'] == "grayscale" and day_night_info[-1]['time of day'] == "daytime":
            processing_count+=1
            print("Processing queue ", processing_count)
            print("Currently processing image:", temp_filename, "\n\n")
            # sky border point comparison
            sky_boundary_edge, sky_filled, labelled_img = sky_line_border_point(nrow, ncol, sky_avg_mean, avg_mean, r_layer)
            
            highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename = combined_operations(sky_boundary_edge, sky_filled, labelled_img, temp_filename, temp_filename_only, highest_acc_mask, highest_acc_value, highest_acc_skyline, highest_acc_filename)
        
        # else -> any images classified as night time
        else:
            print("Skip processing this image!!!\nConditions not suitable for processing\n\n")
            
            
            # get mask of highest accuracy reading (during program runtime)
            print("Best mask accuracy obtained from this program's current iteration: ", highest_acc_value, "\n", "Mask referenced from: ", highest_acc_filename, "\n")
                        
            # 
            pp.figure(temp_filename)
            pp.subplot(1,2,1)
            # title_name = "Ground truth - ref from: " + highest_acc_filename
            pp.title("Ground truth")
            pp.imshow(highest_acc_mask, cmap='gray')
            # pp.set_xlabel("ref from: " + highest_acc_filename)
            pp.subplot(1,2,2)
            # title_name2 = "Skyline - ref from: " + highest_acc_filename
            pp.title("Skyline")
            pp.imshow(highest_acc_skyline, cmap='gray')
            # pp.set_xlabel("ref from: " + highest_acc_filename)
            saved_img_name = output_folder + "/" + temp_filename_only + ".png"
            pp.savefig(saved_img_name)
            
            # save the resulting outputs to a folder
            saved_img_name1 = output_folder + "/" + temp_filename_only
            # os.mkdir(saved_img_name1)
            if not os.path.exists(saved_img_name1): # if not created
                os.mkdir(saved_img_name1)
            else: # if ald created
                shutil.rmtree(saved_img_name1)
                os.mkdir(saved_img_name1)
            saved_img_name2 = saved_img_name1 + "/Ground truth.png"
            saved_img_name3 = saved_img_name1 + "/Skyline.png"
            cv.imwrite(saved_img_name2, highest_acc_mask)
            cv.imwrite(saved_img_name3, highest_acc_skyline)
            
            
            
    # outside of looping each image  
    
    # show the overall accuracy of sky detection for the dataset
    acc1 = 0
    for i in range(0, len(overall_accuracy)):
        acc1+=overall_accuracy[i]
    acc1 = acc1/len(overall_accuracy)
    print("Overall Accuracy reading of (Ground truth/mask): ", acc1, "\n################## NOTE #################", "\nAccuracy of images classfied as night time by the algorithm will not be included in the accuracy reading result as show above!!\nJustification: mean intensity of these images are too dull and edges are near invisible" , "\n\n")
    
    

    # nicer view of day night info
    # create text file for saving record
    text_file_path = output_folder + "/TimeOfDayInfo.txt"
    timeofday_file = open(text_file_path, "a")

    index = 1
    print("Processed information on image taken during what time of day -")
    timeofday_file.write("Processed information on image taken during what time of day -\n")
    for i in day_night_info:
        if index < 10:
            print(index, ".  ", i['file name'], " ", i['img type']," ", i['time of day'])
            timeofday_file.write(str(index) + ".  " + i['file name'] + " " + i['img type'] +" " + i['time of day'] + "\n")
        else:
            print(index, ". ", i['file name'], " ", i['img type']," ", i['time of day'])
            timeofday_file.write(str(index) + ". " + i['file name'] + " " + i['img type'] +" " + i['time of day'] + "\n")
        index+=1
        
    timeofday_file.close()

# program begins here
# ---------------------------------------------------------------------------
file_scanner = os.scandir()
folders = []

# scan for all folders in current dir
for entry in file_scanner:
    if entry.is_dir():
        folders.append(str(entry.name))
user_choice = input("Which dataset do you wish to proceed? (please give only the dataset number as ur input)\nChoice: ")
is_found = False
for fol in folders:
    if fol == user_choice:
        is_found = True
if is_found:
    # create file path to read the provided mask
    path = user_choice + "/*.*"
    mask_path = "mask/" + user_choice + ".png"
    print("\n\n")
    # create directory to store the outputs from the algo
    new_dir = user_choice + " output"
    if not os.path.exists(new_dir): # if not created
        os.mkdir(new_dir)
    else: # if ald created
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)
    # call sky detection algo
    path = user_choice + "/*.*"
    # mask_path = "mask/" + user_choice + ".png"
    sky_region_detection(path, new_dir, mask_path)
    print("Program finished executing...")
else:
    print("No such dataset/folder found, program terminated...")
    sys.exit()