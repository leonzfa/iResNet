#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Spatial Transformer Network (Max Jaderberg etc, Spatial Transformer Networks.)
 * current version: affine transform + bilinear sampling
 */
template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
 public:
  explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SpatialTransformer"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; };
  virtual inline int MinTopBlobs() const { return 1; }
 
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> target_; // 1*3*(H*W), different channels and nums have the same coordinate system
  Blob<Dtype> source_; // N*(2*H*W), channels are shared
  Blob<int> source_range_; // N*H*W*4
  int width_;
  int height_;
  int channel_;
  int num_;
  
  int map_size_;
  
  // gpu only
  Blob<Dtype> source_grad_cache_;//  C*N*2*H*W to allow separately handling gradients for different channels 
  Blob<Dtype> source_grad_op_; // C*1 to sum over difference channels
};

}  // namespace caffe

#endif  // CAFFE_CUSTOM_LAYERS_HPP_
