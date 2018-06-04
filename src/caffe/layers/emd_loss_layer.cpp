#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/emd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EMDLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);  
  
  // convert ground truth to many channels
  many_bottom_vec_.clear();
  many_bottom_vec_.push_back(bottom[1]);
  many_top_vec_.clear();
  many_top_vec_.push_back(&many_);
  LayerParameter many_param;
  const int output_channels = bottom[0]->channels();
  many_param.mutable_onetomany_param()->set_output_channels(output_channels);
  many_layer_.reset(new OneToManyLayer<Dtype>(many_param));
  many_layer_->SetUp(many_bottom_vec_, many_top_vec_);  
}

template <typename Dtype>
void EMDLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  many_layer_->Reshape(many_bottom_vec_, many_top_vec_); 
 
  predict_cdf_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
 
  gt_cdf_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
 
  many_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  inv_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  inv_diff_cdf_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EMDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void EMDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EMDLossLayer);
#endif

INSTANTIATE_CLASS(EMDLossLayer);
REGISTER_LAYER_CLASS(EMDLoss);

}  // namespace caffe
