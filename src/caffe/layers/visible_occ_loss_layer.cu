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

template <typename Dtype>
void VisibleOccLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  Dtype dot, loss; 
  normalize_coeff_ = bottom[0]->count() / bottom[0]->channels();

  E1_warp_layer_->Forward(E1_warp_bottom_vec_,E1_warp_top_vec_); // IL_warp = warp(IR,DL)
//    std::cout<<"E1_warp_layer_"<< "Forwarded"<<std::endl;
  E1_F1_layer_->Forward(E1_F1_bottom_vec_, E1_F1_top_vec_);        // F1 = IL - IL_warp
//    std::cout<<"E1_F1_layer_"<< "Forwarded"<<std::endl;
  E1_F2_layer_->Forward(E1_F1_top_vec_, E1_F2_top_vec_);           // F2 = F1^2
//    std::cout<<"E1_F2_layer_"<< "Forwarded"<<std::endl;
  E1_sF2_layer_->Forward(E1_F2_top_vec_, E1_sF2_top_vec_);           // F2 = F1^2
//    std::cout<<"E1_sF2_layer_"<< "Forwarded"<<std::endl;
  E1_OF_layer_->Forward(E1_OF_bottom_vec_, E1_OF_top_vec_);        // OF = OL * F2
//    std::cout<<"E1_OF_layer_"<< "Forwarded"<<std::endl;
  E1_elts_layer_->Forward(E1_elts_bottom_vec_, E1_elts_top_vec_); // elts = F2 - OL * F2
//    std::cout<<"E1_elts_layer_"<< "Forwarded"<<std::endl;
  E1_eltp_layer_->Forward(E1_eltp_bottom_vec_, E1_eltp_top_vec_);  // eltp = eta_o * (F2 - OL * F2)
//    std::cout<<"E1_eltp_layer_"<< "Forwarded"<<std::endl;

  E2_warp_self_layer_->Forward(E2_warp_self_bottom_vec_,E2_warp_self_top_vec_);  // W = warp_self(DR)
//    std::cout<<"E2_warp_self_layer_"<< "Forwarded"<<std::endl;
  E2_elts_layer_->Forward(E2_elts_bottom_vec_, E2_elts_top_vec_); // elts = OL - W
//    std::cout<<"E2_elts_layer_"<< "Forwarded"<<std::endl;
  E2_abs_layer_->Forward(E2_elts_top_vec_, E2_abs_top_vec_);       // abs = |OL - W|
//    std::cout<<"E2_abs_layer_"<< "Forwarded"<<std::endl;
  E2_eltp_layer_->Forward(E2_eltp_bottom_vec_, E2_eltp_top_vec_);  // eltp = beta_w * |OL - W|
//    std::cout<<"E2_eltp_layer_"<< "Forwarded"<<std::endl;

  E3_stn_layer_->Forward(E3_stn_bottom_vec_, E3_stn_top_vec_);
//    std::cout<<"E3_stn_layer_"<< "Forwarded"<<std::endl;
  E3_elts_layer_->Forward(E3_elts_bottom_vec_, E3_elts_top_vec_);
//    std::cout<<"E3_elts_layer_"<< "Forwarded"<<std::endl;
  E3_abs_layer_->Forward(E3_elts_top_vec_, E3_abs_top_vec_);
//    std::cout<<"E3_abs_layer_"<< "Forwarded"<<std::endl;
  E3_eltp_layer_->Forward(E3_eltp_bottom_vec_, E3_eltp_top_vec_);
//    std::cout<<"E3_eltp_layer_"<< "Forwarded"<<std::endl;

  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
//  std::cout<<"sum_layer_"<< "Forwarded"<<std::endl;

  caffe_gpu_dot(sum_output_.count(), sum_output_.gpu_data(), ones_.gpu_data(), &dot);
  loss = dot / normalize_coeff_;
  top[0]->mutable_cpu_data()[0] = loss;
  // for test
  Dtype E1_sums, E1_sum, E2_sum, E3_sum;
  caffe_gpu_dot(E1_elts_output_.count(), E1_elts_output_.gpu_data(), ones_.gpu_data(), &E1_sums);
  caffe_gpu_dot(E1_eltp_output_.count(), E1_eltp_output_.gpu_data(), ones_.gpu_data(), &E1_sum);
  caffe_gpu_dot(E2_eltp_output_.count(), E2_eltp_output_.gpu_data(), ones_.gpu_data(), &E2_sum);
  caffe_gpu_dot(E3_eltp_output_.count(), E3_eltp_output_.gpu_data(), ones_.gpu_data(), &E3_sum);
  std::cout<<"E1s = "<<E1_sums<<std::endl;
  std::cout<<"E1p = "<<E1_sum<<std::endl;
  std::cout<<"E2 = "<<E2_sum<<std::endl;
  std::cout<<"E3 = "<<E3_sum<<std::endl;
  // test E2
  Dtype E2_ws,E2_elts,E2_abs;
  caffe_gpu_dot(E2_warp_self_output_.count(), E2_warp_self_output_.gpu_data(), ones_.gpu_data(), &E2_ws);
  caffe_gpu_dot(E2_elts_output_.count(), E2_elts_output_.gpu_data(), ones_.gpu_data(), &E2_elts);
  caffe_gpu_dot(E2_abs_output_.count(), E2_abs_output_.gpu_data(), ones_.gpu_data(), &E2_abs);
  std::cout<<"test E2 ..."<<std::endl;
  std::cout<<"E2_ws = "<<E2_ws<<std::endl;
  std::cout<<"E2_elts = "<<E2_elts<<std::endl;
  std::cout<<"E2_abs = "<<E2_abs<<std::endl;
  //test E3
  Dtype E3_stn,E3_elts,E3_abs;
  caffe_gpu_dot(E3_stn_output_.count(), E3_stn_output_.gpu_data(), ones_.gpu_data(), &E3_stn);
  caffe_gpu_dot(E3_elts_output_.count(), E3_elts_output_.gpu_data(), ones_.gpu_data(), &E3_elts);
  caffe_gpu_dot(E3_abs_output_.count(), E3_abs_output_.gpu_data(), ones_.gpu_data(), &E3_abs);
  std::cout<<"test E3 ..."<<std::endl;
  std::cout<<"E3_stn = "<<E3_stn<<std::endl;
  std::cout<<"E3_elts = "<<E3_elts<<std::endl;
  std::cout<<"E3_abs = "<<E3_abs<<std::endl;
}

template <typename Dtype>
void VisibleOccLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
//  std::cout<<"Backward_gpu for VisibleOccLoss"<<std::endl;
  const Dtype alpha = top[0]->cpu_diff()[0] / normalize_coeff_;
  vector<bool> prop_down(1,true);

  caffe_gpu_axpby(sum_output_.count(), alpha, ones_.gpu_data(),
      Dtype(0), sum_output_.mutable_gpu_diff());

  sum_layer_->Backward(sum_bottom_vec_, prop_down, sum_top_vec_);
//    std::cout<<"sum_layer_"<<" Backwarded"<<std::endl;

  E3_eltp_layer_->Backward(E3_eltp_top_vec_, prop_down, E3_eltp_bottom_vec_);
//    std::cout<<"E3_eltp_layer_"<<" Backwarded"<<std::endl;
  E3_abs_layer_->Backward(E3_abs_top_vec_, prop_down, E3_elts_top_vec_);
//    std::cout<<"E3_abs_layer_"<<" Backwarded"<<std::endl;
  E3_elts_layer_->Backward(E3_elts_top_vec_, prop_down, E3_elts_bottom_vec_);
//    std::cout<<"E3_elts_layer_"<<" Backwarded"<<std::endl;
  E3_stn_layer_->Backward(E3_stn_top_vec_, prop_down, E3_stn_bottom_vec_);
//    std::cout<<"E3_stn_layer_"<<" Backwarded"<<std::endl;

  E2_eltp_layer_->Backward(E2_eltp_top_vec_, prop_down, E2_eltp_bottom_vec_);  // eltp = beta_w * |OL - W|
//    std::cout<<"E2_eltp_layer_"<<" Backwarded"<<std::endl;
  E2_abs_layer_->Backward(E2_abs_top_vec_, prop_down, E2_elts_top_vec_);       // abs = |OL - W|
//    std::cout<<"E2_abs_layer_"<<" Backwarded"<<std::endl;
  E2_elts_layer_->Backward(E2_elts_top_vec_, prop_down, E2_elts_bottom_vec_); // elts = OL - W
//    std::cout<<"E2_elts_layer_"<<" Backwarded"<<std::endl;
  E2_warp_self_layer_->Backward(E2_warp_self_top_vec_, prop_down, E2_warp_self_bottom_vec_);  // W = warp_self(DR)
//    std::cout<<"E2_warp_self_layer_"<<" Backwarded"<<std::endl;

  E1_eltp_layer_->Backward(E1_eltp_top_vec_, prop_down, E1_eltp_bottom_vec_);  // eltp = eta_o * (F2 - OL * F2)
//    std::cout<<"E1_eltp_layer_"<<" Backwarded"<<std::endl;
  E1_elts_layer_->Backward(E1_elts_top_vec_, prop_down, E1_elts_bottom_vec_); // elts = F2 - OL * F2
//    std::cout<<"E1_elts_layer_"<<" Backwarded"<<std::endl;
  E1_OF_layer_->Backward(E1_OF_top_vec_, prop_down, E1_OF_bottom_vec_);        // OF = OL * F2
//    std::cout<<"E1_OF_layer_"<<" Backwarded"<<std::endl;
  E1_sF2_layer_->Backward(E1_sF2_top_vec_,prop_down, E1_F2_top_vec_);           // F2 = F1^2
//    std::cout<<"E1_sF2_layer_"<<" Backwarded"<<std::endl;
  E1_F2_layer_->Backward(E1_F2_top_vec_, prop_down, E1_F1_top_vec_);           // F2 = F1^2
//    std::cout<<"E1_F2_layer_"<<" Backwarded"<<std::endl;
  E1_F1_layer_->Backward(E1_F1_top_vec_, prop_down, E1_F1_bottom_vec_);        // F1 = IL - IL_warp
//    std::cout<<"E1_F1_layer_"<<" Backwarded"<<std::endl;
  E1_warp_layer_->Backward(E1_warp_top_vec_, prop_down, E1_warp_bottom_vec_); // IL_warp = warp(IR,DL)
//    std::cout<<"E1_warp_layer_"<<" Backwarded"<<std::endl;

}

INSTANTIATE_LAYER_GPU_FUNCS(VisibleOccLossLayer);

}  // namespace caffe
