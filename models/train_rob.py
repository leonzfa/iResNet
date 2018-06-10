#!/usr/bin/env python
import os, sys
import subprocess

caffe_bin = '/home/leo/caffe_lzf/build/tools/caffe'

# =========================================================

if not os.path.isfile(caffe_bin):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)

print('args:', sys.argv[1:])

args = [caffe_bin, 'train', '-solver', '../ROB_training/solver_rob_stage_one.prototxt', '-gpu', '0'] + sys.argv[1:]
cmd = str.join(' ', args)
print('Executing %s' % cmd)

subprocess.call(args)
