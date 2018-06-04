#ifndef VISIBLE_OCC_LOSS_LAYER_HPP_
#define VISIBLE_OCC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/warp_layer.hpp"
#include "caffe/layers/warp_self_layer.hpp"
#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/st_layer.hpp"

namespace caffe {


/**
 * Visible loss by Zhengfa
 */

//Forward declare
template <typename Dtype> class ConvolutionLayer;
template <typename Dtype> class EltwiseLayer;
template <typename Dtype> class AbsValLayer;
template <typename Dtype> class WarpDisparityLayer;
template <typename Dtype> class WarpSelfLayer;
template <typename Dtype> class PowerLayer;
template <typename Dtype> class SpatialTransformerLayer;

template <typename Dtype>
class VisibleOccLossLayer : public LossLayer<Dtype> {
 public:
  explicit VisibleOccLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VisibleOccLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 5; }
  virtual inline int MaxBottomBlobs() const { return 5; }

 protected:
  /// @copydoc VisibleOccLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> ones_;
  Blob<Dtype> eta_o_;
  Blob<Dtype> beta_w_;
  Blob<Dtype> beta_o_;
  Dtype normalize_coeff_;
  //const vector<int>* bottom_shape_;
  // param
  Dtype param_eta_o;
  Dtype param_beta_w;
  Dtype param_beta_o;

  // layers
  shared_ptr<WarpDisparityLayer<Dtype> > E1_warp_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E1_F1_layer_;
  shared_ptr<PowerLayer<Dtype> > E1_F2_layer_;
  shared_ptr<ConvolutionLayer<Dtype> > E1_sF2_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E1_OF_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E1_elts_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E1_eltp_layer_;

  shared_ptr<WarpSelfLayer<Dtype> > E2_warp_self_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E2_elts_layer_;
  shared_ptr<AbsValLayer<Dtype> > E2_abs_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E2_eltp_layer_;

  shared_ptr<SpatialTransformerLayer<Dtype> > E3_stn_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E3_elts_layer_;
  shared_ptr<AbsValLayer<Dtype> > E3_abs_layer_;
  shared_ptr<EltwiseLayer<Dtype> > E3_eltp_layer_;

  shared_ptr<EltwiseLayer<Dtype> > sum_layer_;


  //vectors and blobs
  vector<Blob<Dtype>*> E1_warp_bottom_vec_;
  vector<Blob<Dtype>*> E1_warp_top_vec_;
  vector<Blob<Dtype>*> E1_F1_bottom_vec_;
  vector<Blob<Dtype>*> E1_F1_top_vec_;
  vector<Blob<Dtype>*> E1_F2_top_vec_;
  vector<Blob<Dtype>*> E1_sF2_top_vec_;
  vector<Blob<Dtype>*> E1_OF_bottom_vec_;
  vector<Blob<Dtype>*> E1_OF_top_vec_;
  vector<Blob<Dtype>*> E1_elts_bottom_vec_;
  vector<Blob<Dtype>*> E1_elts_top_vec_;
  vector<Blob<Dtype>*> E1_eltp_bottom_vec_;
  vector<Blob<Dtype>*> E1_eltp_top_vec_;
  Blob<Dtype> E1_warp_output_;
  Blob<Dtype> E1_F1_output_;
  Blob<Dtype> E1_F2_output_;
  Blob<Dtype> E1_sF2_output_;
  Blob<Dtype> E1_OF_output_;
  Blob<Dtype> E1_elts_output_;
  Blob<Dtype> E1_eltp_output_;

  vector<Blob<Dtype>*> E2_warp_self_bottom_vec_;
  vector<Blob<Dtype>*> E2_warp_self_top_vec_;
  vector<Blob<Dtype>*> E2_elts_bottom_vec_;
  vector<Blob<Dtype>*> E2_elts_top_vec_;
  vector<Blob<Dtype>*> E2_abs_top_vec_;
  vector<Blob<Dtype>*> E2_eltp_bottom_vec_;
  vector<Blob<Dtype>*> E2_eltp_top_vec_;
  Blob<Dtype> E2_warp_self_output_;
  Blob<Dtype> E2_elts_output_;
  Blob<Dtype> E2_abs_output_;
  Blob<Dtype> E2_eltp_output_;

  vector<Blob<Dtype>*> E3_stn_bottom_vec_;
  vector<Blob<Dtype>*> E3_stn_top_vec_;
  vector<Blob<Dtype>*> E3_elts_bottom_vec_;
  vector<Blob<Dtype>*> E3_elts_top_vec_;
  vector<Blob<Dtype>*> E3_abs_top_vec_;
  vector<Blob<Dtype>*> E3_eltp_bottom_vec_;
  vector<Blob<Dtype>*> E3_eltp_top_vec_;
  Blob<Dtype> E3_stn_output_;
  Blob<Dtype> E3_elts_output_;
  Blob<Dtype> E3_abs_output_;
  Blob<Dtype> E3_eltp_output_;

  vector<Blob<Dtype>*> sum_bottom_vec_;
  vector<Blob<Dtype>*> sum_top_vec_;
  Blob<Dtype> sum_output_;
  Blob<Dtype> theta_;
};

}  // namespace caffe

#endif  // VISIBLE_OCC_LOSS_LAYER_HPP_
