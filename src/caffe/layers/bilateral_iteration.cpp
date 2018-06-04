#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/bilateral_iteration.hpp"
#include <iostream>
using namespace std;

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */


template <typename Dtype>
void BilateralIteration<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int direction = this->layer_param_.bi_iter_param().direction();
  int displacement = this->layer_param_.bi_iter_param().displacement();

  //**** update the propagation results
  // weights' channel from 1 to N

  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[1]);

  copy_output_.ReshapeLike(*bottom[0]);
  propagation_shift_.ReshapeLike(*bottom[0]);
  prodx_output_.ReshapeLike(*bottom[0]);
  weights_shift_.ReshapeLike(*bottom[1]);



  copy_bottom_vec_.clear();
  copy_bottom_vec_.push_back(bottom[1]); // one channel to bottom[0]->channels
  copy_top_vec_.clear();
  copy_top_vec_.push_back(&copy_output_);
  LayerParameter copy_param;
  copy_param.mutable_convolution_param()->set_num_output(bottom[0]->channels());
  copy_param.mutable_convolution_param()->add_kernel_size(1);
  copy_param.mutable_convolution_param()->mutable_weight_filler()->set_type("constant");
  copy_param.mutable_convolution_param()->mutable_weight_filler()->set_value(Dtype(1));
  //copy_param.mutable_param()->mutable_lr_mult()->set_value(Dtype(0));
  //copy_param.mutable_param()->mutable_decay_mult()->set_value(Dtype(0));
  copy_layer_.reset(new ConvolutionLayer<Dtype>(copy_param));
  copy_layer_->SetUp(copy_bottom_vec_, copy_top_vec_);
  //
  shiftx_bottom_vec_.clear();
  shiftx_bottom_vec_.push_back(bottom[0]);

  shiftx_top_vec_.clear();
  shiftx_top_vec_.push_back(&propagation_shift_);

  LayerParameter shiftx_param;
  shiftx_param.mutable_shift_param()->set_displacement(int(pow(2,displacement)));
  shiftx_param.mutable_shift_param()->set_direction(direction);
  shiftx_layer_.reset(new ShiftLayer<Dtype>(shiftx_param));
  shiftx_layer_->SetUp(shiftx_bottom_vec_, shiftx_top_vec_);
    
  prodx_bottom_vec_.clear();
  prodx_bottom_vec_.push_back(&copy_output_);
  prodx_bottom_vec_.push_back(&propagation_shift_);
    
  prodx_top_vec_.clear();
  prodx_top_vec_.push_back(&prodx_output_);
    
  cout<<"shape of copy_output_"<<endl;
  cout<<copy_output_.num()<<endl;
  cout<<copy_output_.channels()<<endl;
  cout<<copy_output_.height()<<endl;
  cout<<copy_output_.width()<<endl;

  cout<<"shape of propagation_shift_"<<endl;
  cout<<propagation_shift_.num()<<endl;
  cout<<propagation_shift_.channels()<<endl;
  cout<<propagation_shift_.height()<<endl;
  cout<<propagation_shift_.width()<<endl;

  LayerParameter prodx_param;
  prodx_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  prodx_layer_.reset(new EltwiseLayer<Dtype>(prodx_param));
  prodx_layer_->SetUp(prodx_bottom_vec_, prodx_top_vec_);
    LOG(INFO) << ("-------------");
  //update
  update_bottom_vec_.clear();
  update_bottom_vec_.push_back(bottom[0]);
  update_bottom_vec_.push_back(&prodx_output_);
    

  cout<<"shape of bottom[1]"<<endl;
  cout<<bottom[1]->num()<<endl;
  cout<<bottom[1]->channels()<<endl;
  cout<<bottom[1]->height()<<endl;
  cout<<bottom[1]->width()<<endl;

  cout<<"shape of prodx_output_"<<endl;
  cout<<prodx_output_.num()<<endl;
  cout<<prodx_output_.channels()<<endl;
  cout<<prodx_output_.height()<<endl;
  cout<<prodx_output_.width()<<endl;

  update_top_vec_.clear();
  update_top_vec_.push_back(top[0]);
  LayerParameter update_param;
  update_param.mutable_eltwise_param()->add_coeff(1.);
  update_param.mutable_eltwise_param()->add_coeff(1.);
  update_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  update_layer_.reset(new EltwiseLayer<Dtype>(update_param));
  update_layer_->SetUp(update_bottom_vec_, update_top_vec_);
    
  //**** calculate the weights
  shiftw_bottom_vec_.clear();
  shiftw_bottom_vec_.push_back(bottom[1]);
    
  shiftw_top_vec_.clear();
  shiftw_top_vec_.push_back(&weights_shift_);
  
  LayerParameter shiftw_param;
  shiftw_param.mutable_shift_param()->set_displacement(int(pow(2,displacement)));
  shiftw_param.mutable_shift_param()->set_direction(direction);
  shiftw_layer_.reset(new ShiftLayer<Dtype>(shiftw_param));
  shiftw_layer_->SetUp(shiftw_bottom_vec_, shiftw_top_vec_);

  // Sum layer configuration
  prodw_bottom_vec_.clear();
  prodw_bottom_vec_.push_back(bottom[1]);
  prodw_bottom_vec_.push_back(&weights_shift_);

  prodw_top_vec_.clear();
  prodw_top_vec_.push_back(top[1]);

  LayerParameter prodw_param;
  prodw_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  prodw_layer_.reset(new EltwiseLayer<Dtype>(prodw_param));
  prodw_layer_->SetUp(prodw_bottom_vec_, prodw_top_vec_);


}

template <typename Dtype>
void BilateralIteration<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
return;
}



/**
 * Forward pass during the inference.
 */

template <typename Dtype>
void BilateralIteration<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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


template<typename Dtype>
void BilateralIteration<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_CLASS(BilateralIteration);
}  // namespace caffe
