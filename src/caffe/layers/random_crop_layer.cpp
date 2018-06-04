#include <cfloat>
#include <vector>

#include "caffe/layers/random_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/downsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK(bottom[0]->channels() == 7) << "Random Crop Layer's bottom channel is 7";
  
  target_height_ = this->layer_param_.random_crop_param().target_height();
  target_width_  = this->layer_param_.random_crop_param().target_width();

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), target_height_, target_width_);  
  offsets = vector<int>(2, 0);  
}

template <typename Dtype>
void RandomCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  return;
}

template <typename Dtype>
void RandomCropLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  return;
}

template <typename Dtype>
void RandomCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(RandomCropLayer);
#endif

INSTANTIATE_CLASS(RandomCropLayer);
REGISTER_LAYER_CLASS(RandomCrop);

}  // namespace caffe
