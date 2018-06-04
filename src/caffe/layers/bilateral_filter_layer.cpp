#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bilateral_filter_layer.hpp"

#include <cmath>

namespace caffe {
template <typename Dtype>
void BilateralFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //


  diff_left_.ReshapeLike(*bottom[0]);
  diff_right_.ReshapeLike(*bottom[0]);
  diff_up_.ReshapeLike(*bottom[0]);
  diff_bottom_.ReshapeLike(*bottom[0]);

  dl_abs_.ReshapeLike(*bottom[0]);
  dr_abs_.ReshapeLike(*bottom[0]);
  du_abs_.ReshapeLike(*bottom[0]);
  db_abs_.ReshapeLike(*bottom[0]);



  dl_abs_sum_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
  dr_abs_sum_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
  du_abs_sum_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
  db_abs_sum_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());

  horizontal_output_.ReshapeLike(*bottom[0]);


  // used in backward
  dl_data_sign_.ReshapeLike(*bottom[0]);
  dr_data_sign_.ReshapeLike(*bottom[0]);
  du_data_sign_.ReshapeLike(*bottom[0]);
  db_data_sign_.ReshapeLike(*bottom[0]);
  local_left_diff_.ReshapeLike(*bottom[0]);
  local_right_diff_.ReshapeLike(*bottom[0]);
  local_up_diff_.ReshapeLike(*bottom[0]);
  local_bottom_diff_.ReshapeLike(*bottom[0]);




  num_iterations_ = this->layer_param_.bilateral_filter_param().iteration();

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  
  iteration_left_blobs_.resize(num_iterations_ );    // cache output
  iteration_right_blobs_.resize(num_iterations_ );   // cache output
  iteration_up_blobs_.resize(num_iterations_ );      // cache output
  iteration_bottom_blobs_.resize(num_iterations_ );  // cache output

  weights_left_blobs_.resize(num_iterations_ );   // weights
  weights_right_blobs_.resize(num_iterations_ );  // weights
  weights_up_blobs_.resize(num_iterations_ );     // weights
  weights_bottom_blobs_.resize(num_iterations_ ); // weights
  
  // allocate
  LOG(INFO) << ("1");
  for (int i = 0; i < num_iterations_; ++i) {
    // left
    iteration_left_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
    weights_left_blobs_[i].reset(new Blob<Dtype>(num_, 1, height_, width_));
    // right
    iteration_right_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
    weights_right_blobs_[i].reset(new Blob<Dtype>(num_, 1, height_, width_));
    // up
    iteration_up_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
    weights_up_blobs_[i].reset(new Blob<Dtype>(num_, 1, height_, width_));
    // bottom
    iteration_bottom_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
    weights_bottom_blobs_[i].reset(new Blob<Dtype>(num_, 1, height_, width_));
  }

LOG(INFO) << ("2");
  // if i == 0, bottom[0] + shift_bottom[0] * exp_output_ -> iteration_output_blobs_[0]
  // else, iteration_output_blobs_[i-1] + shift_iteration_output_blobs_[i-1] * weights[i] -> iteration_output_blobs_[i]



  // 


  bilateral_iterations_left_bottom_vec_   = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_right_bottom_vec_  = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_up_bottom_vec_     = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_bottom_bottom_vec_ = new vector<Blob<Dtype>*>[num_iterations_];

  bilateral_iterations_left_top_vec_ = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_right_top_vec_ = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_up_top_vec_ = new vector<Blob<Dtype>*>[num_iterations_];
  bilateral_iterations_bottom_top_vec_ = new vector<Blob<Dtype>*>[num_iterations_];
LOG(INFO) << ("3");


  // left and right
bilateral_iterations_left_.resize(num_iterations_);
bilateral_iterations_right_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    // left
    //bilateral_iterations_left_[i].reset(new BilateralIteration<Dtype>());

    //// curr_x and curr_w
    bilateral_iterations_left_bottom_vec_[i].clear();
    if(i==0){
      bilateral_iterations_left_bottom_vec_[i].push_back(bottom[0]);
      bilateral_iterations_left_bottom_vec_[i].push_back(&dl_abs_sum_);
    }
    else{
      bilateral_iterations_left_bottom_vec_[i].push_back(iteration_left_blobs_[i-1].get());
      bilateral_iterations_left_bottom_vec_[i].push_back(weights_left_blobs_[i-1].get());
    }
LOG(INFO) << ("31");
    //// next_x
    bilateral_iterations_left_top_vec_[i].clear();
    bilateral_iterations_left_top_vec_[i].push_back(iteration_left_blobs_[i].get());
    bilateral_iterations_left_top_vec_[i].push_back(weights_left_blobs_[i].get());
  LOG(INFO) << ("32");  
    LayerParameter bi_iter_left_param;
    bi_iter_left_param.mutable_bi_iter_param()->set_direction(1);
    bi_iter_left_param.mutable_bi_iter_param()->set_displacement(i);
  LOG(INFO) << ("33");  
    bilateral_iterations_left_[i].reset(new BilateralIteration<Dtype>(bi_iter_left_param));
  LOG(INFO) << ("34");  
    bilateral_iterations_left_[i]->SetUp(bilateral_iterations_left_bottom_vec_[i], bilateral_iterations_left_top_vec_[i]);
LOG(INFO) << ("35");
    // right
    //// curr_x and curr_w
    bilateral_iterations_right_bottom_vec_[i].clear();
    if(i==0){
      bilateral_iterations_right_bottom_vec_[i].push_back(bottom[0]);
      bilateral_iterations_right_bottom_vec_[i].push_back(&dr_abs_sum_);
    }
    else{
      bilateral_iterations_right_bottom_vec_[i].push_back(iteration_right_blobs_[i-1].get());
      bilateral_iterations_right_bottom_vec_[i].push_back(weights_right_blobs_[i-1].get());
    }
    //// next_x
    bilateral_iterations_right_top_vec_[i].clear();
    bilateral_iterations_right_top_vec_[i].push_back(iteration_right_blobs_[i].get());
    bilateral_iterations_right_top_vec_[i].push_back(weights_right_blobs_[i].get());
   
    LayerParameter bi_iter_right_param;
    bi_iter_right_param.mutable_bi_iter_param()->set_direction(1);
    bi_iter_right_param.mutable_bi_iter_param()->set_displacement(i);
    bilateral_iterations_right_[i].reset(new BilateralIteration<Dtype>(bi_iter_right_param));
    bilateral_iterations_right_[i]->SetUp(bilateral_iterations_right_bottom_vec_[i], bilateral_iterations_right_top_vec_[i]);
  }
LOG(INFO) << ("4");
  horizontal_bottom_vec_.clear();
  horizontal_bottom_vec_.push_back(iteration_left_blobs_[num_iterations_ - 1].get());
  horizontal_bottom_vec_.push_back(iteration_right_blobs_[num_iterations_ - 1].get());
  horizontal_bottom_vec_.push_back(bottom[0]);
  horizontal_top_vec_.clear();
  horizontal_top_vec_.push_back(&horizontal_output_);

  LayerParameter horizontal_param;
  horizontal_param.mutable_eltwise_param()->add_coeff(1.);
  horizontal_param.mutable_eltwise_param()->add_coeff(1.);
  horizontal_param.mutable_eltwise_param()->add_coeff(-1.);
  horizontal_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  horizontal_layer_.reset(new EltwiseLayer<Dtype>(horizontal_param));
  horizontal_layer_->SetUp(horizontal_bottom_vec_, horizontal_top_vec_);
LOG(INFO) << ("5");
  // up and bottom
bilateral_iterations_up_.resize(num_iterations_);
bilateral_iterations_bottom_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    // up
    //// curr_x and curr_w
    bilateral_iterations_up_bottom_vec_[i].clear();
    if(i==0){
      bilateral_iterations_up_bottom_vec_[i].push_back(&horizontal_output_);
      bilateral_iterations_up_bottom_vec_[i].push_back(&du_abs_sum_);
    }
    else{
      bilateral_iterations_up_bottom_vec_[i].push_back(iteration_up_blobs_[i-1].get());
      bilateral_iterations_up_bottom_vec_[i].push_back(weights_up_blobs_[i-1].get());
    }
    //// next_x
    bilateral_iterations_up_top_vec_[i].clear();
    bilateral_iterations_up_top_vec_[i].push_back(iteration_up_blobs_[i].get());
    bilateral_iterations_up_top_vec_[i].push_back(weights_up_blobs_[i].get());

    LayerParameter bi_iter_up_param;
    bi_iter_up_param.mutable_bi_iter_param()->set_direction(1);
    bi_iter_up_param.mutable_bi_iter_param()->set_displacement(i);
    bilateral_iterations_up_[i].reset(new BilateralIteration<Dtype>(bi_iter_up_param));
    bilateral_iterations_up_[i]->SetUp(bilateral_iterations_up_bottom_vec_[i], bilateral_iterations_up_top_vec_[i]);

    // bottom
    //bilateral_iterations_bottom_[i].reset(new BilateralIteration<Dtype>());

    bilateral_iterations_bottom_bottom_vec_[i].clear();
    if(i==0){
      bilateral_iterations_bottom_bottom_vec_[i].push_back(&horizontal_output_);
      bilateral_iterations_bottom_bottom_vec_[i].push_back(&db_abs_sum_);
    }
    else{
      bilateral_iterations_bottom_bottom_vec_[i].push_back(iteration_bottom_blobs_[i-1].get());
      bilateral_iterations_bottom_bottom_vec_[i].push_back(weights_bottom_blobs_[i-1].get());
    }
    //// next_x
    bilateral_iterations_bottom_top_vec_[i].clear();
    bilateral_iterations_bottom_top_vec_[i].push_back(iteration_bottom_blobs_[i].get());
    bilateral_iterations_bottom_top_vec_[i].push_back(weights_bottom_blobs_[i].get());

LOG(INFO) << ("56");   
    LayerParameter bi_iter_bottom_param;
    bi_iter_bottom_param.mutable_bi_iter_param()->set_direction(1);
    bi_iter_bottom_param.mutable_bi_iter_param()->set_displacement(i);
    bilateral_iterations_bottom_[i].reset(new BilateralIteration<Dtype>(bi_iter_bottom_param));
    bilateral_iterations_bottom_[i]->SetUp(bilateral_iterations_bottom_bottom_vec_[i], bilateral_iterations_bottom_top_vec_[i]);
  }

  LOG(INFO) << ("6");
  top[0]->Reshape(num_, channels_, height_, width_);

  vertical_bottom_vec_.clear();
  vertical_bottom_vec_.push_back(iteration_up_blobs_[num_iterations_ - 1].get());
  vertical_bottom_vec_.push_back(iteration_bottom_blobs_[num_iterations_ - 1].get());
  vertical_bottom_vec_.push_back(&horizontal_output_);
  vertical_top_vec_.clear();
  vertical_top_vec_.push_back(top[0]);

  LayerParameter vertical_param;
  vertical_param.mutable_eltwise_param()->add_coeff(1.);
  vertical_param.mutable_eltwise_param()->add_coeff(1.);
  vertical_param.mutable_eltwise_param()->add_coeff(-1.);
  vertical_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  vertical_layer_.reset(new EltwiseLayer<Dtype>(vertical_param));
  vertical_layer_->SetUp(vertical_bottom_vec_, vertical_top_vec_);

  this->param_propagate_down_.resize(this->blobs_.size(), true);

  LOG(INFO) << ("BilateralLayer initialized.");
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
return;
}


template <typename Dtype>
void BilateralFilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
return;
}


template <typename Dtype>
void BilateralFilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 return;
}




INSTANTIATE_CLASS(BilateralFilterLayer);
REGISTER_LAYER_CLASS(BilateralFilter);

}  // namespace caffe
