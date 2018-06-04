#include <cfloat>
#include <vector>

#include "caffe/layers/error_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
__global__ void FindNotNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? Dtype(1) : Dtype(0);
  }
}

template <typename Dtype>
__global__ void KillNaNs(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index]==in[index] ? in[index] : Dtype(0);
  }
}

template <typename Dtype>
__global__ void KillMasked(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(0.5) ? out[index] : Dtype(0);
  }
}


template <typename Dtype>
__global__ void MaskPlateauValues(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
    if(fabs(in[index]) < plateau) out[index] = Dtype(0); // Mask out plateau values and keep other as is
  }
}

template <typename Dtype>
__global__ void MaskPlateauValuesBinary(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
      if(in[index] == in[index]){
	      if(in[index] > plateau)
		  out[index] = Dtype(1);
	      else if(in[index] < -plateau)
		  out[index] = Dtype(1);
	      else
		  out[index] = Dtype(0);
      }
      else
           out[index] = Dtype(2);
  }
}

template <typename Dtype>
__global__ void MaskValid(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
      if(in[index] == in[index]){
	      if(in[index] > plateau)
		  out[index] = Dtype(0);
	      else if(in[index] < -plateau)
		  out[index] = Dtype(0);
	      else
		  out[index] = Dtype(1);
      }
      else
           out[index] = Dtype(0); // invalid
  }
}


template <typename Dtype>
__global__ void MaskPlateauValuesTernary(const int n, const Dtype* in, Dtype* out, Dtype plateau) {
  CUDA_KERNEL_LOOP(index, n) {
      if(in[index] > plateau)
          out[index] = Dtype(1);
      else if(in[index] < -plateau)
          out[index] = Dtype(-1);
      else
          out[index] = Dtype(0);
  }
}

template <typename Dtype>
void ErrorLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Blob<Dtype> *diffptr = top[0];
    //std::cout<<"error Forward_gpu"<<std::endl;
    diff_layer_->Forward(bottom, top);
    //std::cout<<"error Forward_gpu OK"<<std::endl;
    // find not-NaNs
    //FindNotNaNs<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
     //     diffptr->count(), diffptr->gpu_data(), mask_.mutable_gpu_data());
    //cudaDeviceSynchronize();
    //CUDA_POST_KERNEL_CHECK;
    //std::cout<<"find not-NaNs"<<std::endl;

    // kill NaNs
    //KillMasked<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
    //      diffptr->count(), mask_.gpu_data(), diffptr->mutable_gpu_data());
    //cudaDeviceSynchronize();
    //CUDA_POST_KERNEL_CHECK;

    //std::cout<<"kill masked"<<std::endl;

    if(error_type_ == 0){
        MaskPlateauValuesBinary<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
            diffptr->count(), diffptr->mutable_gpu_data(), diffptr->mutable_gpu_data(), plateau_);
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
    }
    else if(error_type_ == 1){
        MaskPlateauValuesTernary<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
            diffptr->count(), diffptr->mutable_gpu_data(), diffptr->mutable_gpu_data(), plateau_);
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
    }
    else{ // especially used for 
        MaskValid<Dtype><<<CAFFE_GET_BLOCKS(diffptr->count()), CAFFE_CUDA_NUM_THREADS>>>(
            diffptr->count(), diffptr->mutable_gpu_data(), diffptr->mutable_gpu_data(), plateau_);
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
    }

    //std::cout<<"forward done!"<<std::endl;
}


template <typename Dtype>
void ErrorLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //std::cout<<"backward gpu"<<std::endl;
    return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ErrorLabelLayer);

}  // namespace caffe
