#ifndef CAFFE_RANDOM_CROP_LAYER_HPP_
#define CAFFE_RANDOM_CROP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RandomCropLayer : public Layer<Dtype> {

 public:
  explicit RandomCropLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "RandomCropLayer";
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  //virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 protected:

  int target_height_;
  int target_width_;
  vector<int> offsets;

};


}  // namespace caffe

#endif  // CAFFE_RANDOM_CROP_LAYER_HPP_
