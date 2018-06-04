#!/usr/bin/env python
import os, sys
import subprocess
import time
from dateutil import parser
import datetime
from math import ceil
from math import floor
from util_stereo import *

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)

# You need to change this directories according to you own setting.
CAFFE_ROOT  = '/home/leo/caffe_lzf'


# =========================================================
caffe_bin    = os.path.join(CAFFE_ROOT,'build/tools/caffe.bin')
img_size_bin = os.path.join(CAFFE_ROOT,'build/tools/get_image_size')
dataset_dir  = os.path.join(CAFFE_ROOT,'data/datasets_middlebury2014')
submit_dir   = 'submission_results'
template = 'model/deploy_iResNet_ROB.tpl.prototxt'
template_mirror = 'model/deploy_iResNet_ROB_mirror.tpl.prototxt'
subprocess.call('mkdir -p tmp', shell=True)

method = 'iResNet_ROB'
MAX_SIZE = 1034496.
W_FACTOR = 1.4
H_FACTOR = 1.5
DIVISOR  = 64.
LOG_FILE = 'log.txt'
GPU_ID = 0


def get_image_size(filename):
    global img_size_bin
    dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([img_size_bin, filename])).split(',')]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list


def sizes_equal(size1, size2):
    return size1[0] == size2[0] and size1[1] == size2[1]


if not (os.path.isfile(caffe_bin) and os.path.isfile(img_size_bin)):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)

# Converts a string to bytes (for writing the string into a file). Provided for
# compatibility with Python 2 and 3.
def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')


def CalculateTime(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    started = 0
    for curr_line in lines:
        # start time
        if curr_line.find('Running for 1 iterations') != -1:
           idx1 = curr_line.find(' ')
           curr_line2 = curr_line[idx1+1:]
           idx2 = curr_line2.find(' ')
           start_time_string = curr_line[idx1+1:idx1+idx2+1]
           start_time = parser.parse(start_time_string)
           started = 1
        if curr_line.find('pfmwriter_layer.cpp') != -1 and started == 1:
           idx1 = curr_line.find(' ')
           curr_line2 = curr_line[idx1+1:]
           idx2 = curr_line2.find(' ')
           end_time_string = curr_line[idx1+1:idx1+idx2+1]
           end_time = parser.parse(end_time_string)
           break
    sec = (end_time-start_time).total_seconds()
    os.remove(log_file)
    return sec




def RunMethod(caffemodel, submit_dir, dataset_dir, dataset_name, istraining):
    if ~os.path.exists(submit_dir):
        subprocess.call('mkdir -p ' + submit_dir, shell=True)

    file_names = os.listdir(dataset_dir)
    for name in file_names:

        if name[0] != dataset_name[0]:
            continue

        if dataset_name == 'ETH3D2017':
            target_folder = os.path.join(submit_dir, 'low_res_two_view')
            disparity_name = name[10:]
            disparity_name_mirror = []
        elif dataset_name == 'Middlebury2014':
            sub_folder = 'trainingH' if istraining else 'testH'
            target_folder = os.path.join(submit_dir, dataset_name + '_' + method, sub_folder, name[15:])
            disparity_name = 'disp0' + method
            disparity_name_mirror = 'disp0' + method + '_s'
        elif dataset_name == 'Kitti2015':
            target_folder = os.path.join(submit_dir, 'disp_0_pfm')
            final_folder = os.path.join(submit_dir, 'disp_0')
            disparity_name = name[10:]
            disparity_name_mirror = []
 

        # read images
        im0 = os.path.join(dataset_dir,name,'im0.png')
        im1 = os.path.join(dataset_dir,name,'im1.png')

        im0_size = get_image_size(im0)
        im1_size = get_image_size(im1)
    
        with open('tmp/img1.txt', "w") as tfile:
            tfile.write("%s\n" % im0)

        with open('tmp/img2.txt', "w") as tfile:
            tfile.write("%s\n" % im1)    
 
        width  = im0_size[0]
        height = im0_size[1]
        
        # resize the images
        adapted_width = ceil(W_FACTOR * width / DIVISOR) * DIVISOR
        adapted_height = ceil(H_FACTOR * height / DIVISOR) * DIVISOR
        adapted_size = adapted_width * adapted_height
        if adapted_size > MAX_SIZE:
            scaling = (MAX_SIZE / adapted_size) ** 0.5
            adapted_width = floor(scaling * adapted_width / DIVISOR) * DIVISOR
            adapted_height_tmp = ceil(scaling * adapted_height / DIVISOR) * DIVISOR
            if adapted_width * adapted_height_tmp > MAX_SIZE:
                adapted_height = floor(scaling * adapted_height / DIVISOR) * DIVISOR
            else:
                adapted_height = ceil(scaling * adapted_height / DIVISOR) * DIVISOR

        print 'Image size = %d' % (adapted_height*adapted_width)
        rescale_coeff_x = width / adapted_width
        

        # deploy.prototxt
        replacement_list = {
            '$ADAPTED_WIDTH': ('%d' % adapted_width),
            '$ADAPTED_HEIGHT': ('%d' % adapted_height),
            '$TARGET_WIDTH': ('%d' % width),
            '$TARGET_HEIGHT': ('%d' % height),
            '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
            '$TARGET_FOLDER':('%s' % target_folder),
            '$METHOD_NAME':('%s' % disparity_name),
            '$METHOD_NAME_MIRROR':('%s' % disparity_name_mirror),
        }
            
        proto = ''
        with open(template, "r") as tfile:
            proto = tfile.read()
        for r in replacement_list:
            proto = proto.replace(r, replacement_list[r])
        with open('tmp/deploy.prototxt', "w") as tfile:
            tfile.write(proto)  

        # Run caffe            
        myargs = [caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
            '-weights', caffemodel,
            '-iterations', '1',
            '-gpu', '0']

        cmd = str.join(' ', myargs)
        cmd = cmd + ' 2>&1 | tee ' + LOG_FILE
        print('Executing %s' % cmd)
        os.popen(cmd).read()            
        print('** The resulting disparity is stored in %s.pfm' % os.path.join(target_folder, disparity_name))

        runtime = CalculateTime(LOG_FILE)
        if dataset_name == 'ETH3D2017':
            runtime = 'runtime %.2f' % runtime
            time_output_path = os.path.join(target_folder, disparity_name + '.txt')
        else:
            runtime = '%.2f' % runtime
            time_output_path = os.path.join(target_folder, 'time' + method + '.txt')

        with open(time_output_path, 'wb') as time_file:
            time_file.write(StrToBytes(runtime))    
        print('** The resulting time file is stored in %s.txt\n' % os.path.join(target_folder, disparity_name))


        # Run caffe for Middlebury2014 one more time to calculate right disparity
        if dataset_name == 'Middlebury2014':
            proto = ''
            with open(template_mirror, "r") as tfile:
                proto = tfile.read()
            for r in replacement_list:
                proto = proto.replace(r, replacement_list[r])
            with open('tmp/deploy_mirror.prototxt', "w") as tfile:
                tfile.write(proto)  

            myargs = [caffe_bin, 'test', '-model', 'tmp/deploy_mirror.prototxt',
                '-weights', caffemodel,
                '-iterations', '1',
                '-gpu', '0']

            cmd = str.join(' ', myargs)
            cmd = cmd + ' 2>&1 | tee ' + LOG_FILE
            print('Executing %s' % cmd)

            os.popen(cmd).read()      
            print('** The resulting disparity is stored in %s.pfm\n' % os.path.join(target_folder, disparity_name_mirror))
            os.remove(LOG_FILE)
        
        # convert kitti format
        if dataset_name == 'Kitti2015':
            if ~os.path.exists(final_folder):
                subprocess.call('mkdir -p ' + final_folder, shell=True)

            src_pfm_path = os.path.join(target_folder, disparity_name) + '.pfm'
            dest_png_path = os.path.join(final_folder, disparity_name) + '.png'
            ConvertMiddlebury2014PfmToKitti2015Png(src_pfm_path, dest_png_path)




if __name__ == '__main__':  

    caffemodel = sys.argv[1]    
    training_dataset_dir = os.path.join(dataset_dir,'training')
    test_dataset_dir = os.path.join(dataset_dir,'test')  

    # test eth3d
    dataset_name = 'ETH3D2017'
    # training set    
    RunMethod(caffemodel, submit_dir, training_dataset_dir, dataset_name, True)
    # test set    
    RunMethod(caffemodel, submit_dir, test_dataset_dir, dataset_name, False)

    # test middlebury
    dataset_name = 'Middlebury2014'
    # training set    
    RunMethod(caffemodel, submit_dir, training_dataset_dir, dataset_name, True)
    # test set    
    RunMethod(caffemodel, submit_dir, test_dataset_dir, dataset_name, False)
  

    # test kitti
    dataset_name = 'Kitti2015'
    # test set    
    RunMethod(caffemodel, submit_dir, test_dataset_dir, dataset_name, False)


