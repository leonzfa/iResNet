#!/bin/bash 


CAFFE_ROOT/build/tools/convert_imageset_and_disparity.bin ./FlyingThings3D_release/FlyingThings3D_release_TRAIN_clean.list ./datasets_lmdbs/FlyingThings3D_release_TRAIN_clean_lmdb 0 lmdb
CAFFE_ROOT/build/tools/convert_imageset_and_disparity.bin ./FlyingThings3D_release/FlyingThings3D_release_TEST_clean.list ./datasets_lmdbs/FlyingThings3D_release_TEST_clean_lmdb 0 lmdb

CAFFE_ROOT/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_middle.list ./datasets_lmdbs/middlebury_lmdb 0 lmdb
CAFFE_ROOT/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_kitti.list ./datasets_lmdbs/kitti_lmdb 0 lmdb
CAFFE_ROOT/build/tools/convert_imageset_and_disparity.bin ./datasets_lmdbs/rob_eth3d.list ./datasets_lmdbs/eth3d_lmdb 0 lmdb
