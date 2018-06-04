#include <algorithm>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/data_switch_layer.hpp"

namespace caffe {

template <typename Dtype>
void DataSwitchLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {	
 /*  
  a_cycle_ = this->layer_param().data_switch_param().a_cycle();
  b_cycle_ = this->layer_param().data_switch_param().b_cycle();
  count_ = 0;
  */
}

template <typename Dtype>
void DataSwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DataSwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  /*
  const int total_cycle = a_cycle_ + b_cycle_;
  const int iter = count_ % total_cycle;
  if( iter < a_cycle_) {
	caffe_copy<Dtype>(top[0]->count(),bottom[0]->mutable_cpu_data(),top[0]->mutable_cpu_data());
  }
  else {
	caffe_copy<Dtype>(top[0]->count(),bottom[1]->mutable_cpu_data(),top[0]->mutable_cpu_data());
  }
  count_++;*/
  return;
}

template <typename Dtype>
void DataSwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {	
  return;
}

#ifdef CPU_ONLY
STUB_GPU(DataSwitchLayer);
#endif

INSTANTIATE_CLASS(DataSwitchLayer);
REGISTER_LAYER_CLASS(DataSwitch);

}  // namespace caffe
