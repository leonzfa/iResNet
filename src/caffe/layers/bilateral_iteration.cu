#include <vector>

#include "caffe/util/io.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bilateral_iteration.hpp"


namespace caffe {

template <typename Dtype>
void BilateralIteration<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {	
  // propagation
  copy_layer_->Forward(copy_bottom_vec_, copy_top_vec_);
  shiftx_layer_->Forward(shiftx_bottom_vec_, shiftx_top_vec_);
  prodx_layer_->Forward(prodx_bottom_vec_, prodx_top_vec_);
  update_layer_->Forward(update_bottom_vec_, update_top_vec_);

  // calculate weights
  shiftw_layer_->Forward(shiftw_bottom_vec_, shiftw_top_vec_);
  prodw_layer_->Forward(prodw_bottom_vec_, prodw_top_vec_);

}



template <typename Dtype>
void BilateralIteration<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> prop_down(1,true);
  // calculate weights
  prodw_layer_->Backward(prodw_bottom_vec_, prop_down, prodw_top_vec_);
  shiftw_layer_->Backward(shiftw_bottom_vec_, prop_down, shiftw_top_vec_);

  // propagation
  update_layer_->Backward(update_top_vec_, prop_down, update_bottom_vec_);
  prodx_layer_->Backward(prodx_top_vec_, prop_down, prodx_bottom_vec_);
  shiftx_layer_->Backward(shiftx_top_vec_, prop_down, shiftx_bottom_vec_);
  copy_layer_->Backward(copy_top_vec_, prop_down, copy_bottom_vec_);
}

INSTANTIATE_LAYER_GPU_FUNCS(BilateralIteration);

}	// namespace caffe
