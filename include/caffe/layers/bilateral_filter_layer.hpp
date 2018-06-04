#ifndef CAFFE_BILATERAL_FILTER_LAYER_HPP_
#define CAFFE_BILATERAL_FILTER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bilateral_iteration.hpp"
#include "caffe/layers/eltwise_layer.hpp"
//#include "caffe/layers/conv_layer.hpp"
//#include "caffe/layers/shift_layer.hpp"

namespace caffe {
//declare
template <typename Dtype> class BilateralIteration;
template <typename Dtype> class EltwiseLayer;

template <typename Dtype>
class BilateralFilterLayer : public Layer<Dtype> {

 public:
  explicit BilateralFilterLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "BilateralFilter";
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 protected:

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;


  int num_iterations_;



  Blob<Dtype> diff_left_;
  Blob<Dtype> diff_right_;
  Blob<Dtype> diff_up_;
  Blob<Dtype> diff_bottom_;

  Blob<Dtype> dl_abs_;
  Blob<Dtype> dr_abs_;
  Blob<Dtype> du_abs_;
  Blob<Dtype> db_abs_;

  Blob<Dtype> dl_abs_sum_;
  Blob<Dtype> dr_abs_sum_;
  Blob<Dtype> du_abs_sum_;
  Blob<Dtype> db_abs_sum_;

  Blob<Dtype> horizontal_output_;


  // used in backward
  Blob<Dtype> dl_data_sign_;
  Blob<Dtype> dr_data_sign_;
  Blob<Dtype> du_data_sign_;
  Blob<Dtype> db_data_sign_;
  Blob<Dtype> local_left_diff_;
  Blob<Dtype> local_right_diff_;
  Blob<Dtype> local_up_diff_;
  Blob<Dtype> local_bottom_diff_;


  // bottom vec
  vector<Blob<Dtype>*> *bilateral_iterations_left_bottom_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_right_bottom_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_up_bottom_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_bottom_bottom_vec_;
  // top vec
  vector<Blob<Dtype>*> *bilateral_iterations_left_top_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_right_top_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_up_top_vec_;
  vector<Blob<Dtype>*> *bilateral_iterations_bottom_top_vec_;

  vector<Blob<Dtype>*> horizontal_bottom_vec_;
  vector<Blob<Dtype>*> horizontal_top_vec_;
  vector<Blob<Dtype>*> vertical_bottom_vec_;
  vector<Blob<Dtype>*> vertical_top_vec_;

  vector<shared_ptr<Blob<Dtype> > > iteration_left_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_right_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_up_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_bottom_blobs_;

  vector<shared_ptr<Blob<Dtype> > > weights_left_blobs_;
  vector<shared_ptr<Blob<Dtype> > > weights_right_blobs_;
  vector<shared_ptr<Blob<Dtype> > > weights_up_blobs_;
  vector<shared_ptr<Blob<Dtype> > > weights_bottom_blobs_;

  // BilateralIteration
  vector<shared_ptr<BilateralIteration<Dtype> > > bilateral_iterations_left_;
  vector<shared_ptr<BilateralIteration<Dtype> > > bilateral_iterations_right_;
  vector<shared_ptr<BilateralIteration<Dtype> > > bilateral_iterations_up_;
  vector<shared_ptr<BilateralIteration<Dtype> > > bilateral_iterations_bottom_;

  shared_ptr<EltwiseLayer<Dtype> > horizontal_layer_;
  shared_ptr<EltwiseLayer<Dtype> > vertical_layer_;
};


}  // namespace caffe

#endif  // CAFFE_MULTI_STAGE_MEANFIELD_LAYER_HPP_
