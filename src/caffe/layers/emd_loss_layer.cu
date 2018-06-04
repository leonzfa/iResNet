#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/emd_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
__global__ void CalCDF(const int nthreads, int N, int C, int H, int W,
		const Dtype* pdf, Dtype* cdf) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % W;
		const int s = (index / W) % H;		
		const int i = index / (W * H);

		Dtype tmp = 0;
		for( int j = 0; j < C; j++){
			tmp += pdf[i * (C * H * W) + j * (H * W) + W * s + t];
			cdf[i * (C * H * W) + j * (H * W) + W * s + t] = tmp;
		}
  }
}


//*************************
//* @brief:
//* in:  N1HW
//* out: NCHW
//* If found nan in the groundtruth, then the values in mask_ along the channels wil be 0, which means the gradient will not backpropagate.
//*************************
template <typename Dtype>
__global__ void CalNanMask(const int nthreads, int N, int C, int H, int W, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    
    const int t = index % W; // w
    const int s = (index / W) % H; // h		
    const int i = index / (W * H); // n

    if(in[index]==in[index]){
       for(int j = 0; j < C; j++)
          out[i * (C * H * W) + j * (H * W) + W * s + t] = Dtype(1);
    }else{
       for(int j = 0; j < C; j++)
          out[i * (C * H * W) + j * (H * W) + W * s + t] = Dtype(0);
    }
  }
} 


template <typename Dtype>
__global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
  }
}


template <typename Dtype>
void EMDLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  
  // calculate many_
  Blob<Dtype> *manyptr = many_top_vec_[0]; // many channels
  many_layer_ ->Forward(many_bottom_vec_, many_top_vec_);
  // calculate mask for nan
  const int W = bottom[0]->width();
  const int H = bottom[0]->height();
  const int C = bottom[0]->channels();
  const int N = bottom[0]->num();  
  int nthreads  = bottom[1]->count(); // ground truth
  int bnthreads = bottom[0]->count();
  CalNanMask<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,bottom[1]->gpu_data(), mask_.mutable_gpu_data());

  // calculate cdf for prediction, then mask nan
  CalCDF<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,bottom[0]->gpu_data(), predict_cdf_.mutable_gpu_data());  
  KillMasked<Dtype><<<CAFFE_GET_BLOCKS(bnthreads), CAFFE_CUDA_NUM_THREADS>>>(
        bnthreads, mask_.gpu_data(), predict_cdf_.mutable_gpu_data());

  // calculate cdf for groundtruth, then mask nan
  CalCDF<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,manyptr->gpu_data(), gt_cdf_.mutable_gpu_data());
  KillMasked<Dtype><<<CAFFE_GET_BLOCKS(bnthreads), CAFFE_CUDA_NUM_THREADS>>>(
        bnthreads, mask_.gpu_data(), gt_cdf_.mutable_gpu_data());

  // calculate valid samples
  caffe_gpu_dot(bottom[0]->count(), mask_.gpu_data(), mask_.gpu_data(), &normalize_coeff_forward_);
  normalize_coeff_forward_ /= mask_.channels();
  // LOG(INFO)<<"normalize_coeff_forward_ = "<<normalize_coeff_forward_;

  // calculate emd loss
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      predict_cdf_.gpu_data(),
      gt_cdf_.gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / normalize_coeff_forward_;
  top[0]->mutable_cpu_data()[0] = loss;  
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void InverseChannels(const int nthreads, int N, int C, int H, int W,
		const Dtype* pos, Dtype* neg) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % W;
		const int s = (index / W) % H;		
		const int i = index / (W * H);
		for( int j = 0; j < C; j++){
                        neg[i * (C * H * W) + (C-j-1) * (H * W) + W * s + t] = pos[i * (C * H * W) + j * (H * W) + W * s + t];			
		}
  }
}



template <typename Dtype>
void EMDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{  
  bool prop_down = propagate_down[0];
  if(bottom.size() > 1) prop_down |= propagate_down[1];
  
  //Blob<Dtype> *diffptr = diff_top_vec_[0];
  
  if (prop_down) {

    // diff_ = predict_cdf - gt_cdf
    // inv_diff
    const int W = bottom[0]->width();
    const int H = bottom[0]->height();
    const int C = bottom[0]->channels();
    const int N = bottom[0]->num(); 
    int nthreads = bottom[1]->count(); // ground truth
    int bnthreads = bottom[0]->count(); // ground truth
    InverseChannels<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,diff_.gpu_data(), inv_diff_.mutable_gpu_data());
    
    CalCDF<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,inv_diff_.gpu_data(), inv_diff_cdf_.mutable_gpu_data());

    InverseChannels<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N,C,H,W,inv_diff_cdf_.gpu_data(), bottom[0]->mutable_gpu_diff());

    // mask nan
    KillMasked<Dtype><<<CAFFE_GET_BLOCKS(bnthreads), CAFFE_CUDA_NUM_THREADS>>>(
        bnthreads, mask_.gpu_data(), bottom[0]->mutable_gpu_diff());

    return;
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(EMDLossLayer);

}  // namespace caffe
