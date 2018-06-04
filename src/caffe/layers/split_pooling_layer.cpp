#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/split_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

// This layer is modified from pooling layer. It divides the feature maps from N*C*H*W to 4N*C*(H/2)*(W/2).

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SplitPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  // should check if the H and W are even
  kernel_h_ = 2;
  kernel_w_ = 2;
  pad_h_ = 0;
  pad_w_ = 0;
  stride_h_ = 2;
  stride_w_ = 2;
}

template <typename Dtype>
void SplitPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();  

 
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  
  top[0]->Reshape(bottom[0]->num(), channels_ * 4, pooled_height_,
      pooled_width_);   
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void SplitPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // The count of top[0] equals to that of bottom[0]: top[0]->count() == bottom[0]->count()
  const int top_count = top[0]->count();  
  const int count_quarter = static_cast<int>(ceil(static_cast<float>(top[0]->count() / 4)));

  // initialize
  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0;
  }
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
	for (int pw = 0; pw < pooled_width_; ++pw) {
	  // calculate the cordinates at the bottom[0]
	  int hstart = ph * stride_h_ - pad_h_;
	  int wstart = pw * stride_w_ - pad_w_;

          hstart = max(hstart, 0);
	  wstart = max(wstart, 0);
	  
	  top_data[0 * count_quarter + ph * pooled_width_ + pw] = bottom_data[hstart * width_ + wstart];
	  top_data[1 * count_quarter + ph * pooled_width_ + pw] = bottom_data[hstart * width_ + wstart + 1];
	  top_data[2 * count_quarter + ph * pooled_width_ + pw] = bottom_data[(hstart + 1) * width_ + wstart];
	  top_data[3 * count_quarter + ph * pooled_width_ + pw] = bottom_data[(hstart + 1) * width_ + wstart + 1];	
	}
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }    
}

template <typename Dtype>
void SplitPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top.size() > 1;
  //const int* mask = NULL;  // suppress warnings about uninitialized variables
  //const Dtype* top_mask = NULL;
  const int count_quarter = static_cast<int>(ceil(static_cast<float>(bottom[0]->count() / 4)));

    // The main loop
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
	for (int pw = 0; pw < pooled_width_; ++pw) {
	  int hstart = ph * stride_h_ - pad_h_;
	  int wstart = pw * stride_w_ - pad_w_;	  
	  hstart = max(hstart, 0);
	  wstart = max(wstart, 0);
          bottom_diff[hstart * width_ + wstart]           = top_diff[0 * count_quarter + ph * pooled_width_ + pw];
          bottom_diff[hstart * width_ + wstart + 1]       = top_diff[1 * count_quarter + ph * pooled_width_ + pw];
          bottom_diff[(hstart + 1) * width_ + wstart]     = top_diff[2 * count_quarter + ph * pooled_width_ + pw];
          bottom_diff[(hstart + 1) * width_ + wstart + 1] = top_diff[3 * count_quarter + ph * pooled_width_ + pw];
	}
      }
      // offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SplitPoolingLayer);
#endif

INSTANTIATE_CLASS(SplitPoolingLayer);
REGISTER_LAYER_CLASS(SplitPooling);
}  // namespace caffe
