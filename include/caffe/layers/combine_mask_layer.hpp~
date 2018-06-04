#ifndef ERROR_LABEL_LAYER_HPP_
#define ERROR_LABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


#include "caffe/layers/eltwise_layer.hpp"


namespace caffe {


/**
 * Error label layer by Zhengfa
 */

//Forward declare

template <typename Dtype> class EltwiseLayer;

template <typename Dtype>
class ErrorLabelLayer : public Layer<Dtype> {
 public:
  explicit ErrorLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ErrorLable"; }
  virtual inline int MinBottomBlobs() const { return 2; }
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

  Blob<Dtype> mask_;  
  int error_type_;
  Dtype plateau_;

  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<EltwiseLayer<Dtype> > diff_layer_;
  Blob<Dtype> diff_output_;
  vector<Blob<Dtype>*> diff_bottom_vec_;
  vector<Blob<Dtype>*> diff_top_vec_;

};

}  // namespace caffe

#endif  // CONFIDENCE_LOSS_LAYER_HPP_
