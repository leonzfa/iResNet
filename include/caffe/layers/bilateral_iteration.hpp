#ifndef CAFFE_BILATERAL_ITERATION_HPP_
#define CAFFE_BILATERAL_ITERATION_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/shift_layer.hpp"

namespace caffe {

//Forward declare
template <typename Dtype> class EltwiseLayer;
template <typename Dtype> class ShiftLayer;
template <typename Dtype> class ConvolutionLayer;

template <typename Dtype>
class BilateralIteration: public Layer<Dtype> {
 public:
  explicit BilateralIteration(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BilateralIteration"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 

 protected:

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Blob<Dtype> copy_output_;
  Blob<Dtype> propagation_shift_;
  Blob<Dtype> prodx_output_;
  Blob<Dtype> weights_shift_;
  
  vector<Blob<Dtype>*> copy_bottom_vec_;
  vector<Blob<Dtype>*> copy_top_vec_;
  vector<Blob<Dtype>*> shiftx_bottom_vec_;
  vector<Blob<Dtype>*> shiftx_top_vec_;
  vector<Blob<Dtype>*> prodx_bottom_vec_;
  vector<Blob<Dtype>*> prodx_top_vec_;
  vector<Blob<Dtype>*> update_bottom_vec_;
  vector<Blob<Dtype>*> update_top_vec_;
  vector<Blob<Dtype>*> shiftw_bottom_vec_;
  vector<Blob<Dtype>*> shiftw_top_vec_;
  vector<Blob<Dtype>*> prodw_bottom_vec_;
  vector<Blob<Dtype>*> prodw_top_vec_;

  shared_ptr<ConvolutionLayer<Dtype> > copy_layer_;
  shared_ptr<ShiftLayer<Dtype> > shiftx_layer_;
  shared_ptr<EltwiseLayer<Dtype> > prodx_layer_;
  shared_ptr<EltwiseLayer<Dtype> > update_layer_;
  shared_ptr<ShiftLayer<Dtype> > shiftw_layer_;
  shared_ptr<EltwiseLayer<Dtype> > prodw_layer_;

};


}  // namespace caffe

#endif  // CAFFE_BILATERAL_ITERATION_HPP_
