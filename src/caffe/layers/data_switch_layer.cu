#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/data_switch_layer.hpp"
#include "caffe/util/rng.hpp"
#include <ctime>
#include <math.h>




namespace caffe {


template <typename Dtype>
void DataSwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	std::srand ((unsigned)time(NULL));
	  // Random select segmentation point
	  Dtype seg = (static_cast<Dtype>(caffe_rng_rand()) / RAND_MAX);
	  seg = seg - int(seg);
	  int bsize = bottom.size();
	  int out = bsize;

	  for(int i=0; i < bsize; i++){
	    if( seg <= Dtype(i+1) / bsize){		    
	      out = i;
	      break;
	    }
	  }

	 caffe_copy<Dtype>(top[0]->count(),bottom[out]->mutable_gpu_data(),top[0]->mutable_gpu_data());		
	
}

template <typename Dtype>
void DataSwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// caffe_gpu_set<Dtype>(bottom[0]->count(),0,bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(DataSwitchLayer);
}  // namespace caffe
