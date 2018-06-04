import yaml
from sklearn.preprocessing import normalize
import random
import caffe
from caffe.io import caffe_pb2
import numpy as np
import lmdb
import struct
import sys
import os
import time


def parseRecord(val):
        fields = []

        try : 
            datum = caffe_pb2.Datum.FromString(val)
            arr = caffe.io.datum_to_array(datum)
            label = datum.label
            fields = [arr, np.array([label]).reshape((1,1,1))]
        except :
            int_size = 4
            curr_start = 0
            curr_end = 0
            
            while (curr_start < len(val)):
                field_len = struct.unpack('I', val[curr_start : curr_start+int_size])[0]
                curr_start += int_size
                curr_end = curr_start + field_len

                field_datum = caffe_pb2.Datum.FromString(val[curr_start:curr_end])
                field_arr = caffe.io.datum_to_array(field_datum)
                curr_start = curr_end
                fields.append(field_arr)
        return fields

def initialSetup(path):
    keys_labels = dict()
    labels_keys = dict()
    env = lmdb.open(path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, value in cursor:
            datum = caffe_pb2.Datum.FromString(value)
            arr = caffe.io.datum_to_array(datum)
            label = datum.label;
            if not labels_keys.has_key(label): 
                labels_keys[label] = []
            labels_keys[label].append(k) 
            keys_labels[k] = label 
    return keys_labels, labels_keys


#generate batch with predefined number of different classes and images per class
class LabelShufflingLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)       

        self.source_ = layer_params['source']
        self.scales_ = layer_params['scales']
        self.subtract_ = np.zeros_like(self.scales_)
        if 'subtract' in layer_params :  
            self.subtract_ = layer_params['subtract']

        self.batch_size_ = layer_params['batch_size'] #self.num_labels_*self.images_per_label_

        self.max_number_object_per_label_  = np.inf
        if 'max_number_object_per_label' in layer_params : 
            self.max_number_object_per_label_ = layer_params['max_number_object_per_label']

        # structures setup
        self.keys_labels_, self.labels_keys_ = initialSetup(self.source_)
        self.keys_ = self.keys_labels_.keys()
        self.labels_ = self.labels_keys_.keys()
   
        np.random.shuffle(self.labels_)

        self.label_index_ = 0
        self.image_index_ = 0
 
        # figure out the shape
        k = self.keys_[0]

        env = lmdb.open(self.source_, readonly=True) 
        with env.begin() as txn:
            val = txn.get(k) 
        fields = parseRecord(val)
        self.shapes_ = [f.shape for f in fields]
 

    def reshape(self, bottom, top): 
      for i in xrange(len(self.shapes_)):      
            top[i].reshape(self.batch_size_, self.shapes_[i][0], self.shapes_[i][1], self.shapes_[i][2])          

         
    def getBatch(self):
        batch_keys = []

        for i in xrange(self.batch_size_):
            if self.image_index_ >= len(self.labels_keys_[self.labels_[self.label_index_]]) or self.image_index_ >= self.max_number_object_per_label_:
                np.random.shuffle(self.labels_keys_[self.labels_[self.label_index_]])
                self.image_index_ = 0
                self.label_index_+=1    
            if self.label_index_ >= len(self.labels_):
                np.random.shuffle(self.labels_)

                self.label_index_ = 0       
            batch_keys.append(self.labels_keys_[self.labels_[self.label_index_]][self.image_index_])
            self.image_index_+=1
        return batch_keys

    def forward(self, bottom, top):
        batch_data = []        

        for i in xrange(len(self.shapes_)):
            batch_data.append([])

        batch_keys = self.getBatch()
        env = lmdb.open(self.source_, readonly=True) 

        with env.begin() as txn: 
            for k in batch_keys:
                val = txn.get(k)
                fields = parseRecord(val)
                for i in xrange(len(self.shapes_)):
                    batch_data[i].append(fields[i])
               
        for i in xrange(len(self.shapes_)):
            top[i].data[...] = (np.array(batch_data[i]) - self.subtract_[i]) * self.scales_[i]      


    def backward(self, top, propagate_down, bottom):
        pass





