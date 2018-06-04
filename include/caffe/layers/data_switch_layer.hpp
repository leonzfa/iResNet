#ifndef CAFFE_DATA_SWITCH_LAYERS_HPP_
#define CAFFE_DATA_SWITCH_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
//====================================================
// Self defined layers by lzf
//====================================================
  template <typename Dtype>
  class DataSwitchLayer : public Layer<Dtype> {
  public:
    explicit DataSwitchLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "DataSwitchLayer"; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);      
    int a_cycle_;
    int b_cycle_;
    int count_;
  };


}  // namespace caffe

#endif  // CAFFE_DATA_SWITCH_LAYERS_HPP_
