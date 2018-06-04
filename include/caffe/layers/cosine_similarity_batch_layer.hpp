

#ifndef COSINE_SIMILARITY_BATCH_LAYER_HPP_
#define COSINE_SIMILARITY_BATCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {


template <typename Dtype>
class CosineSimilarityBatchLayer : public Layer<Dtype>{
 public:
  explicit CosineSimilarityBatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  virtual inline const char* type() const { return "CosineSimilarityBatch"; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index == 0;
  }
 
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Blob<Dtype> xy_blob;
  
};



}  // namespace caffe

#endif  // COSINE_SIMILARITY_BATCH_LAYER_HPP_



