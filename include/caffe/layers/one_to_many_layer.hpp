#ifndef CAFFE_ONE_TO_MANY_LAYER_HPP_
#define CAFFE_ONE_TO_MANY_LAYER_HPP_

#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the one to many.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class OneToManyLayer : public Layer<Dtype> {
 public:
  explicit OneToManyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OneToMany"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  inline Dtype abs(Dtype x) {
     if(x < 0) return -x; return x;
  }
  inline Dtype max(Dtype x, Dtype y) {
     if(x < y) return y; return x;
  }
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;
  int many_;

  /// scale is used to stored the calculated diff
  Blob<Dtype> scale_;
};

}  // namespace caffe

#endif  // CAFFE_ONE_TO_MANY_LAYER_HPP_
