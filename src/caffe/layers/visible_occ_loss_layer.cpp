#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/visible_occ_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/warp_layer.hpp"
#include "caffe/layers/warp_self_layer.hpp"
#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/st_layer.hpp"

namespace caffe {
//when calculate OCC_L 
//bottom[0]: occL
//E1: CONV(Eltwise_mul(Power(Eltwise_sub(Warp(DL + IR) , IL)),1-OCCL)))
//E2: CONV(ABS(EltWise_sub(WarpSelf(DR),OCCL))
//E3: CONV(ABS(OCCL))
//E_OL = E1 + E2 + E3
//=============
// so we need five bottom blobs
// bottom[0] bottom[1] bottom[2] bottom[3] bottom[4]
// OCCL      IL        IR        DL        DR
//==================================================
// OCCR      IR        IL        DR        DL
// or set a param to indicate



template <typename Dtype>
void VisibleOccLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);


  //implementation of F(s,d_s,I)
  E1_warp_bottom_vec_.clear();
  E1_warp_bottom_vec_.push_back(bottom[2]);
  E1_warp_bottom_vec_.push_back(bottom[3]);
  E1_warp_top_vec_.clear();
  E1_warp_top_vec_.push_back(&E1_warp_output_);
  LayerParameter E1_warp_param;
  E1_warp_layer_.reset(new WarpDisparityLayer<Dtype>(E1_warp_param));
  E1_warp_layer_->SetUp(E1_warp_bottom_vec_,E1_warp_top_vec_);
//  std::cout<<"E1_warp_layer_"<<" has been set up!"<<std::endl;

  E1_F1_bottom_vec_.clear();
  E1_F1_bottom_vec_.push_back(bottom[1]);
  E1_F1_bottom_vec_.push_back(&E1_warp_output_);
  E1_F1_top_vec_.clear();
  E1_F1_top_vec_.push_back(&E1_F1_output_);
  LayerParameter E1_F1_param;
  E1_F1_param.mutable_eltwise_param()->add_coeff(1.);
  E1_F1_param.mutable_eltwise_param()->add_coeff(-1.);
  E1_F1_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  E1_F1_layer_.reset(new EltwiseLayer<Dtype>(E1_F1_param));
  E1_F1_layer_->SetUp(E1_F1_bottom_vec_, E1_F1_top_vec_);
//  std::cout<<"E1_F1_layer_"<<" has been set up!"<<std::endl;

  E1_F2_top_vec_.clear();
  E1_F2_top_vec_.push_back(&E1_F2_output_);
  LayerParameter E1_F2_param;
  E1_F2_param.mutable_power_param()->set_power(Dtype(2));
  E1_F2_layer_.reset(new PowerLayer<Dtype>(E1_F2_param));
  E1_F2_layer_->SetUp(E1_F1_top_vec_, E1_F2_top_vec_);
//  std::cout<<"E1_F2_layer_"<<" has been set up!"<<std::endl;

  E1_sF2_top_vec_.clear();
  E1_sF2_top_vec_.push_back(&E1_sF2_output_);
  LayerParameter E1_sF2_param;
  E1_sF2_param.mutable_convolution_param()->set_num_output(1);
  E1_sF2_param.mutable_convolution_param()->add_kernel_size(1);
  E1_sF2_param.mutable_convolution_param()->mutable_weight_filler()->set_type("constant");
  E1_sF2_param.mutable_convolution_param()->mutable_weight_filler()->set_value(Dtype(1));
  E1_sF2_layer_.reset(new ConvolutionLayer<Dtype>(E1_sF2_param));
  E1_sF2_layer_->SetUp(E1_F2_top_vec_, E1_sF2_top_vec_);
//  std::cout<<"E1_sF2_layer_"<<" has been set up!"<<std::endl;

  //implementation of o_s * F(s,d_s,I)
  E1_OF_bottom_vec_.clear();
  E1_OF_bottom_vec_.push_back(bottom[0]);
  E1_OF_bottom_vec_.push_back(&E1_sF2_output_);
  E1_OF_top_vec_.clear();
  E1_OF_top_vec_.push_back(&E1_OF_output_);
  LayerParameter E1_OF_param;
  E1_OF_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  E1_OF_layer_.reset(new EltwiseLayer<Dtype>(E1_OF_param));
  E1_OF_layer_->SetUp(E1_OF_bottom_vec_, E1_OF_top_vec_);
//  std::cout<<"E1_OF_layer_"<<" has been set up!"<<std::endl;

  //implementation of F(s,d_s,I) - o_s * F(s,d_s,I)
  E1_elts_bottom_vec_.clear();
  E1_elts_bottom_vec_.push_back(&E1_sF2_output_);
  E1_elts_bottom_vec_.push_back(&E1_OF_output_);
  E1_elts_top_vec_.clear();
  E1_elts_top_vec_.push_back(&E1_elts_output_);
  LayerParameter E1_elts_param;
  E1_elts_param.mutable_eltwise_param()->add_coeff(1.);
  E1_elts_param.mutable_eltwise_param()->add_coeff(-1.);
  E1_elts_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  E1_elts_layer_.reset(new EltwiseLayer<Dtype>(E1_elts_param));
  E1_elts_layer_->SetUp(E1_elts_bottom_vec_, E1_elts_top_vec_); //----------------------------------------- to be sumed
//  std::cout<<"E1_elts_layer_"<<" has been set up!"<<std::endl;

  //implementation of o_s * eta_o
  param_eta_o = this->layer_param().visible_occ_loss_param().eta_o();
  //bottom_shape_ = &bottom[0]->shape();
  eta_o_.ReshapeLike(*bottom[0]);
  caffe_set(eta_o_.count(), (Dtype)param_eta_o, eta_o_.mutable_cpu_data());

  E1_eltp_bottom_vec_.clear();
  E1_eltp_bottom_vec_.push_back(&eta_o_);
  E1_eltp_bottom_vec_.push_back(bottom[0]);
  E1_eltp_top_vec_.clear();
  E1_eltp_top_vec_.push_back(&E1_eltp_output_);
  LayerParameter E1_eltp_param;
  E1_eltp_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  E1_eltp_layer_.reset(new EltwiseLayer<Dtype>(E1_eltp_param));
  E1_eltp_layer_->SetUp(E1_eltp_bottom_vec_, E1_eltp_top_vec_); //----------------------------------------- to be sumed
//  std::cout<<"E1_eltp_layer_"<<" has been set up!"<<std::endl;

  //implementation of W(s,D_R)
  E2_warp_self_bottom_vec_.clear();
  E2_warp_self_bottom_vec_.push_back(bottom[4]);
  E2_warp_self_top_vec_.clear();
  E2_warp_self_top_vec_.push_back(&E2_warp_self_output_);
  LayerParameter E2_warp_self_param;
  E2_warp_self_layer_.reset(new WarpSelfLayer<Dtype>(E2_warp_self_param));
  E2_warp_self_layer_->SetUp(E2_warp_self_bottom_vec_,E2_warp_self_top_vec_);
// std::cout<<"E2_warp_self_layer_"<<" has been set up!"<<std::endl;

  //implementation of o_s - W(s,D_R)
  E2_elts_bottom_vec_.clear();
  E2_elts_bottom_vec_.push_back(bottom[0]);
  E2_elts_bottom_vec_.push_back(&E2_warp_self_output_);
  E2_elts_top_vec_.clear();
  E2_elts_top_vec_.push_back(&E2_elts_output_);
  LayerParameter E2_elts_param;
  E2_elts_param.mutable_eltwise_param()->add_coeff(1.);
  E2_elts_param.mutable_eltwise_param()->add_coeff(-1.);
  E2_elts_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  E2_elts_layer_.reset(new EltwiseLayer<Dtype>(E2_elts_param));
  E2_elts_layer_->SetUp(E2_elts_bottom_vec_, E2_elts_top_vec_);
//  std::cout<<"E2_elts_layer_"<<" has been set up!"<<std::endl;

  //implementation of |o_s - W(s,D_R)|
  E2_abs_top_vec_.clear();
  E2_abs_top_vec_.push_back(&E2_abs_output_);
  LayerParameter E2_abs_param;
  E2_abs_layer_.reset(new AbsValLayer<Dtype>(E2_abs_param));
  E2_abs_layer_->SetUp(E2_elts_top_vec_, E2_abs_top_vec_);
//  std::cout<<"E2_abs_layer_"<<" has been set up!"<<std::endl;

  //implementation of |o_s - W(s,D_R)| * beta_w
  param_beta_w = this->layer_param().visible_occ_loss_param().beta_w();
  std::cout<<"param_beta_w = " << param_beta_w<<std::endl;
  beta_w_.ReshapeLike(E2_elts_output_);
  caffe_set(beta_w_.count(), (Dtype)param_beta_w, beta_w_.mutable_cpu_data());

  E2_eltp_bottom_vec_.clear();
  E2_eltp_bottom_vec_.push_back(&beta_w_);
  E2_eltp_bottom_vec_.push_back(&E2_abs_output_); // gao cuo la
  E2_eltp_top_vec_.clear();
  E2_eltp_top_vec_.push_back(&E2_eltp_output_);
  LayerParameter E2_eltp_param;
  E2_eltp_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  E2_eltp_layer_.reset(new EltwiseLayer<Dtype>(E2_eltp_param));
  E2_eltp_layer_->SetUp(E2_eltp_bottom_vec_, E2_eltp_top_vec_); //----------------------------------------- to be sumed
//  std::cout<<"E2_eltp_layer_"<<" has been set up!"<<std::endl;

  //

  vector<int> theta_shape(4);
  theta_shape[0] = bottom[0]->shape(0);
  theta_shape[1] = 1;
  theta_shape[2] = 5;
  theta_shape[3] = 1;
  theta_.Reshape(theta_shape);
  caffe_set(theta_.count(), Dtype(0), theta_.mutable_cpu_data());

  E3_stn_bottom_vec_.clear();
  E3_stn_bottom_vec_.push_back(bottom[0]);
  E3_stn_bottom_vec_.push_back(&theta_);
  E3_stn_top_vec_.clear();
  E3_stn_top_vec_.push_back(&E3_stn_output_);
  LayerParameter E3_stn_param;
  //E3_stn_param.mutable_st_param()->set_theta_1_1(0.0);
  //E3_stn_param.mutable_st_param()->set_theta_1_2(0.0);
  E3_stn_param.mutable_st_param()->set_theta_1_3(-2.0/bottom[0]->shape(3));
  //E3_stn_param.mutable_st_param()->set_theta_2_1(0.0);
  //E3_stn_param.mutable_st_param()->set_theta_2_2(0.0);
  //E3_stn_param.mutable_st_param()->set_theta_2_3(0.0);
  E3_stn_layer_.reset(new SpatialTransformerLayer<Dtype>(E3_stn_param));
  E3_stn_layer_->SetUp(E3_stn_bottom_vec_, E3_stn_top_vec_);
// std::cout<<"E3_stn_layer_"<<" has been set up!"<<std::endl;

  //implementation of o_s - o_t
  E3_elts_bottom_vec_.clear();
  E3_elts_bottom_vec_.push_back(bottom[0]);
  E3_elts_bottom_vec_.push_back(&E3_stn_output_);
  E3_elts_top_vec_.clear();
  E3_elts_top_vec_.push_back(&E3_elts_output_);
  LayerParameter E3_elts_param;
  E3_elts_param.mutable_eltwise_param()->add_coeff(1.);
  E3_elts_param.mutable_eltwise_param()->add_coeff(-1.);
  E3_elts_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  E3_elts_layer_.reset(new EltwiseLayer<Dtype>(E3_elts_param));
  E3_elts_layer_->SetUp(E3_elts_bottom_vec_, E3_elts_top_vec_);
//  std::cout<<"E3_elts_layer_"<<" has been set up!"<<std::endl;

  //implementation of |o_s - o_t|
  E3_abs_top_vec_.clear();
  E3_abs_top_vec_.push_back(&E3_abs_output_);
  LayerParameter E3_abs_param;
  E3_abs_layer_.reset(new AbsValLayer<Dtype>(E3_abs_param));
  E3_abs_layer_->SetUp(E3_elts_top_vec_, E3_abs_top_vec_);
//  std::cout<<"E3_abs_layer_"<<" has been set up!"<<std::endl;

  //implementation of beta_o * |o_s - o_t|
  param_beta_o = this->layer_param().visible_occ_loss_param().beta_o();
  beta_o_.ReshapeLike(*bottom[0]);
  caffe_set(beta_o_.count(), (Dtype)param_beta_o, beta_o_.mutable_cpu_data());

  E3_eltp_bottom_vec_.clear();
  E3_eltp_bottom_vec_.push_back(&beta_o_);
  E3_eltp_bottom_vec_.push_back(&E3_abs_output_);
  E3_eltp_top_vec_.clear();
  E3_eltp_top_vec_.push_back(&E3_eltp_output_);
  LayerParameter E3_eltp_param;
  E3_eltp_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  E3_eltp_layer_.reset(new EltwiseLayer<Dtype>(E3_eltp_param));
  E3_eltp_layer_->SetUp(E3_eltp_bottom_vec_, E3_eltp_top_vec_); //----------------------------------------- to be sumed
//  std::cout<<"E3_eltp_layer_"<<" has been set up!"<<std::endl;

  //sum
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(&E1_elts_output_);
  sum_bottom_vec_.push_back(&E1_eltp_output_);
  sum_bottom_vec_.push_back(&E2_eltp_output_);
  sum_bottom_vec_.push_back(&E3_eltp_output_);
  sum_top_vec_.clear();
  sum_top_vec_.push_back(&sum_output_);
  LayerParameter sum_param;  
  sum_param.mutable_eltwise_param()->add_coeff(0.25);
  sum_param.mutable_eltwise_param()->add_coeff(0.25);
  sum_param.mutable_eltwise_param()->add_coeff(0.25);
  sum_param.mutable_eltwise_param()->add_coeff(0.25);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);
//  std::cout<<"sum_layer_"<<" has been set up!"<<std::endl;

}

template <typename Dtype>
void VisibleOccLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  E1_warp_layer_->Reshape(E1_warp_bottom_vec_,E1_warp_top_vec_); // IL_warp = warp(IR,DL)
 //   std::cout<<"E1_warp_layer_"<<" Reshaped"<<std::endl;
  E1_F1_layer_->Reshape(E1_F1_bottom_vec_, E1_F1_top_vec_);        // F1 = IL - IL_warp
 //   std::cout<<"E1_F1_layer_"<<" Reshaped"<<std::endl;
  E1_F2_layer_->Reshape(E1_F1_top_vec_, E1_F2_top_vec_);           // F2 = F1^2
 //   std::cout<<"E1_F2_layer_"<<" Reshaped"<<std::endl;
  E1_sF2_layer_->Reshape(E1_F2_top_vec_, E1_sF2_top_vec_);           // F2 = F1^2
 //   std::cout<<"E1_sF2_layer_"<<" Reshaped"<<std::endl;
  E1_OF_layer_->Reshape(E1_OF_bottom_vec_, E1_OF_top_vec_);        // OF = OL * F2
 //   std::cout<<"E1_OF_layer_"<<" Reshaped"<<std::endl;
  E1_elts_layer_->Reshape(E1_elts_bottom_vec_, E1_elts_top_vec_); // elts = F2 - OL * F2
 //   std::cout<<"E1_elts_layer_"<<" Reshaped"<<std::endl;
  E1_eltp_layer_->Reshape(E1_eltp_bottom_vec_, E1_eltp_top_vec_);  // eltp = eta_o * (F2 - OL * F2)
 //   std::cout<<"E1_eltp_layer_"<<" Reshaped"<<std::endl;

  E2_warp_self_layer_->Reshape(E2_warp_self_bottom_vec_,E2_warp_self_top_vec_);  // W = warp_self(DR)
 //   std::cout<<"E2_warp_self_layer_"<<" Reshaped"<<std::endl;
  E2_elts_layer_->Reshape(E2_elts_bottom_vec_, E2_elts_top_vec_); // elts = OL - W
 //   std::cout<<"E2_elts_layer_"<<" Reshaped"<<std::endl;
  E2_abs_layer_->Reshape(E2_elts_top_vec_, E2_abs_top_vec_);       // abs = |OL - W|
 //   std::cout<<"E2_abs_layer_"<<" Reshaped"<<std::endl;
  E2_eltp_layer_->Reshape(E2_eltp_bottom_vec_, E2_eltp_top_vec_);  // eltp = beta_w * |OL - W|
 //   std::cout<<"E2_eltp_layer_"<<" Reshaped"<<std::endl;

  E3_stn_layer_->Reshape(E3_stn_bottom_vec_, E3_stn_top_vec_);
 //   std::cout<<"E3_stn_layer_"<<" Reshaped"<<std::endl;
  E3_elts_layer_->Reshape(E3_elts_bottom_vec_, E3_elts_top_vec_);
 //   std::cout<<"E3_elts_layer_"<<" Reshaped"<<std::endl;
  E3_abs_layer_->Reshape(E3_elts_top_vec_, E3_abs_top_vec_);
 //   std::cout<<"E3_abs_layer_"<<" Reshaped"<<std::endl;
  E3_eltp_layer_->Reshape(E3_eltp_bottom_vec_, E3_eltp_top_vec_);
 //   std::cout<<"E3_eltp_layer_"<<" Reshaped"<<std::endl;

  sum_layer_->Reshape(sum_bottom_vec_, sum_top_vec_);
 //   std::cout<<"sum_layer_"<<" Reshaped o"<<std::endl;

  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
  caffe_set(ones_.count()/ones_.channels(), Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
void VisibleOccLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void VisibleOccLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(VisibleOccLossLayer);
#endif

INSTANTIATE_CLASS(VisibleOccLossLayer);
REGISTER_LAYER_CLASS(VisibleOccLoss);

}  // namespace caffe
