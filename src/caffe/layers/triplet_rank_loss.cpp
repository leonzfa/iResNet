#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/triplet_rank_loss.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void TripletRankLossLayer<Dtype>::Reshape(
					    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(1, 1, 1, 1);
  }


  template <typename Dtype>
  Dtype TripletRankLossLayer<Dtype>::Similarity(
						const Dtype* labels, const int i, const int j, const int dim ) {
    Dtype s_sim = 0;
    for(int k=0; k < dim; k++){
      if(labels[i*dim + k] > 0 && labels[i*dim + k] == labels[j*dim + k]){
	s_sim++;
        return s_sim;
      }
    }
    return s_sim;
  }


  template <typename Dtype>
  void TripletRankLossLayer<Dtype>::Forward_cpu(
						const vector<Blob<Dtype>*>& bottom,
						const  vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype margin = this->layer_param_.triplet_param().margin();
    Dtype loss=0;
    Dtype  n_tri = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = i+1; j < num; ++j) {
        Dtype sim_s = Similarity(label, i, j,label_dim );
        for (int k = j+1; k < num; ++k) {
	  Dtype sim_d = Similarity(label, i, k,  label_dim );
	  if(sim_s == sim_d)
	    continue;	
          n_tri++;
	  int a = sim_s > sim_d ? j : k;
	  int b = sim_s > sim_d ? k : j;
	  Dtype norm1=0, norm2 = 0;
	  for(int l=0; l < dim; ++l){
	    norm1 += pow((bottom_data[i*dim + l] - bottom_data[a*dim + l]),2);
	    norm2 += pow((bottom_data[i*dim + l] - bottom_data[b*dim + l]),2);
	  }
	  if(margin + norm1 - norm2 > 0){
	    loss += (margin + norm1 - norm2);
	  }
	}
      }
    }
    
    if(n_tri > 0)
      top[0]->mutable_cpu_data()[0] = loss/n_tri;
    else
      top[0]->mutable_cpu_data()[0] = 0;
  }

  template <typename Dtype>
  void TripletRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    memset(diff, 0, bottom[0]->count()*sizeof(Dtype));
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    int label_dim = bottom[1]->count() / bottom[1]->num();
    Dtype margin = this->layer_param_.triplet_param().margin();
    Dtype  n_tri = 0;

    for (int i = 0; i < num; ++i) {
      for (int j = i+1; j < num; ++j) {
	Dtype sim_s = Similarity(label, i, j,  label_dim );
	for (int k = j+1; k < num; ++k) {
	  Dtype sim_d = Similarity(label, i, k,  label_dim );
	  if(sim_s == sim_d)
	    continue;
	  n_tri++;	
      
	  int a = sim_s > sim_d ? j : k;
	  int b = sim_s > sim_d ? k : j;
	 
	  Dtype norm1=0, norm2 = 0;
	  for(int l=0; l < dim; ++l){
	    norm1 += pow((bottom_data[i*dim + l] - bottom_data[a*dim + l]),2);
	    norm2 += pow((bottom_data[i*dim + l] - bottom_data[b*dim + l]),2);
	  }
	  if(margin +norm1 - norm2 > 0){
	    for(int l=0; l < dim; ++l){
	      diff[i*dim + l] += 2*(bottom_data[b*dim + l] - bottom_data[a*dim + l]);
	      diff[a*dim + l] += 2*(bottom_data[a*dim + l] - bottom_data[i*dim + l]);
	      diff[b*dim + l] += 2*(bottom_data[i*dim + l] - bottom_data[b*dim + l]); 
	    }
	  }
	}
      } 
    }
  
    // Scale down gradient
    if(n_tri > 0)
      caffe_scal(bottom[0]->count(), Dtype(1) / n_tri / margin, diff);

  }


#ifdef CPU_ONLY
    STUB_GPU(TripletRankLossLayer);
#endif

    INSTANTIATE_CLASS(TripletRankLossLayer);
    REGISTER_LAYER_CLASS(TripletRankLoss);

  }  // namespace caffe
