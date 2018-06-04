#ifndef EMD_LOSS_LAYER_HPP_
#define EMD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/one_to_many_layer.hpp"


namespace caffe {


/**
 * EMD loss by Zhengfa Liang
 */

//Forward declare
//template <typename Dtype> class ConvolutionLayer;
template <typename Dtype> class OneToManyLayer;

template <typename Dtype>
class EMDLossLayer : public LossLayer<Dtype> {
 public:
  explicit EMDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EMDLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }

 protected:
  /// @copydoc EMDLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> predict_cdf_, gt_cdf_, mask_, diff_, many_, inv_diff_, inv_diff_cdf_;
  Dtype normalize_coeff_forward_;
  Dtype normalize_coeff_backward_;

  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<OneToManyLayer<Dtype> > many_layer_;
  vector<Blob<Dtype>*> many_top_vec_;
  vector<Blob<Dtype>*> many_bottom_vec_;
  
};

}  // namespace caffe

#endif  // EMD_LOSS_LAYER_HPP_
