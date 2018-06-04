#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/one_to_many_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneToManyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels(); // should be 1
    height_   = bottom[0]->height();
    width_    = bottom[0]->width();
    num_pixels_ = height_ * width_;
    many_     = this->layer_param().onetomany_param().output_channels();
}

template <typename Dtype>
void OneToManyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), many_, height_, width_);
  vector<int> scale_dims = bottom[0]->shape();
  //scale_dims[1] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void OneToManyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // initialize to 0s
  Dtype* V = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), (Dtype)0, V);

  const Dtype* input = bottom[0]->cpu_data();

  // for each input
  Dtype input_value;
  int m, n;
  int idx, flag;
  Dtype wm, wn;

  for(int i = 0; i < num_; ++i) {
      const Dtype*  curr_input = input + (height_ * width_) * i;
      for(int s = 0; s < height_; ++s){
          for(int t = 0; t < width_; ++t) {
              idx = s * width_ + t;
              input_value = abs(curr_input[idx]);
              m  = floor(input_value);
              flag = 0;
              if(m>many_-1){
                  m = many_-1;
                  flag = 1;
              }
              n  = m + 1;
              wm =  max(0, 1 - abs(m - input_value));
              wn =  max(0, 1 - abs(n - input_value));
              V[top[0]->offset(i, m, s, t)] = wm;
              if(flag==0)
                  V[top[0]->offset(i, n, s, t)] = wn;
          }
      }
  }
}

template <typename Dtype>
void OneToManyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  Dtype* scale_data = scale_.mutable_cpu_data();  // s

  if (propagate_down[0]) {
      caffe_copy(top[0]->count(), top_diff, scale_data);
      for (int i = 0; i < num_; ++i) {
        // compute dot(top_diff, top_data) -> scale_data
        for (int k = 0; k < num_pixels_; ++k) {
          scale_data[k] = caffe_cpu_strided_dot<Dtype>(many_,
              top_diff + i * num_pixels_ * many_ + k, num_pixels_,
              top_data + i * num_pixels_ * many_ + k, num_pixels_);
        }
        // subtraction
        caffe_copy(num_pixels_, scale_data, bottom_diff+i*num_pixels_);
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OneToManyLayer);
#endif

INSTANTIATE_CLASS(OneToManyLayer);
REGISTER_LAYER_CLASS(OneToMany);

}  // namespace caffe
