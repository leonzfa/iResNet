#include <cfloat>
#include <vector>

#include "caffe/layers/combine_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CombineMask(const int n, const Dtype* in1, const Dtype* in2, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (in1[index]==Dtype(1) & in2[index]==Dtype(1)) ? Dtype(1) : Dtype(0);
  }
}

template <typename Dtype>
void CombineMaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    Blob<Dtype> *mask = top[0];
    CombineMask<Dtype><<<CAFFE_GET_BLOCKS(mask->count()), CAFFE_CUDA_NUM_THREADS>>>(
        mask->count(), bottom[0]->gpu_data(), bottom[1]->gpu_data(), mask->mutable_gpu_data());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void CombineMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //std::cout<<"backward gpu"<<std::endl;
    return;
}

INSTANTIATE_LAYER_GPU_FUNCS(CombineMaskLayer);
}  // namespace caffe
