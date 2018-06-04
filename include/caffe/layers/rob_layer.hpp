#ifndef CAFFE_ROB_LAYER_HPP_
#define CAFFE_ROB_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/rob_layer.hpp"
#include "caffe/layers/downsample_layer.hpp"

namespace caffe {

template <typename Dtype> class DownsampleLayer;

template <typename Dtype>
class ROBLayer : public Layer<Dtype> {

 public:
  explicit ROBLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "ROBLayer";
  }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 protected:

  int height_half_;
  int width_half_;
  int height_quarter_;
  int width_quarter_;

  int target_height_;
  int target_width_;

  vector<Dtype> coeffs_;
  int num_dataset_;
  vector<int> offsets;
  Blob<unsigned int> rand_vec_;// crop / pad
  Blob<Dtype> seg_vec_;
  Blob<Dtype> mode_vec_;

  // bottom vec
  vector<Blob<Dtype>*> *downsample_left_half_bottom_vec_;
  vector<Blob<Dtype>*> *downsample_right_half_bottom_vec_;
  vector<Blob<Dtype>*> *downsample_disp_half_bottom_vec_;
  vector<Blob<Dtype>*> *downsample_left_quarter_bottom_vec_;
  vector<Blob<Dtype>*> *downsample_right_quarter_bottom_vec_;
  vector<Blob<Dtype>*> *downsample_disp_quarter_bottom_vec_;
  // top vec
  vector<Blob<Dtype>*> *downsample_left_half_top_vec_;
  vector<Blob<Dtype>*> *downsample_right_half_top_vec_;
  vector<Blob<Dtype>*> *downsample_disp_half_top_vec_;
  vector<Blob<Dtype>*> *downsample_left_quarter_top_vec_;
  vector<Blob<Dtype>*> *downsample_right_quarter_top_vec_;
  vector<Blob<Dtype>*> *downsample_disp_quarter_top_vec_;
  
  // blob
  vector<shared_ptr<Blob<Dtype> > > downsampled_left_half_;
  vector<shared_ptr<Blob<Dtype> > > downsampled_right_half_;
  vector<shared_ptr<Blob<Dtype> > > downsampled_disp_half_;

  vector<shared_ptr<Blob<Dtype> > > downsampled_left_quarter_;
  vector<shared_ptr<Blob<Dtype> > > downsampled_right_quarter_;
  vector<shared_ptr<Blob<Dtype> > > downsampled_disp_quarter_;

  vector<shared_ptr<Blob<Dtype> > > bottom_copy_;

  // layer
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_left_half_layer_;
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_left_quarter_layer_;
  
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_right_half_layer_;
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_right_quarter_layer_;
  
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_disp_half_layer_;
  vector<shared_ptr<DownsampleLayer<Dtype> > > downsample_disp_quarter_layer_;  
};


}  // namespace caffe

#endif  // CAFFE_ROB_LAYER_HPP_
