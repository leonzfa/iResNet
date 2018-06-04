 
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/one_to_many_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/benchmark.hpp"


namespace caffe {
// The implementation of kernel_channel_dot is copied from softmax layer
template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}
// nthreads = N*H*W
// U: N*H*W
// V: N*C*H*W
template <typename Dtype>
__global__ void OneToManyForwardGPU(const int nthreads, int N, int C,
                int H, int W, const Dtype* U, Dtype* V) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int t = index % W;
    const int s = (index / W) % H;
    const int i = index / (W * H);

    const Dtype* curr_input = U + i * (H * W); // the channel number of U is 1
    Dtype  input_value = abs(curr_input[s * W + t]);

    int m, n;
    int index_m;
    int index_n;
    Dtype wm;
    Dtype wn;

    m = floor(input_value);

    if(m < C-1){
        n = m + 1;
        wm = max(0.0, 1 - abs(Dtype(m) - input_value));
        wn = max(0.0, 1 - abs(Dtype(n) - input_value));
        index_m = i * (H * W * C) + m * (H * W) + s * W + t;
        index_n = i * (H * W * C) + n * (H * W) + s * W + t;
        V[index_m] = wm;
        V[index_n] = wn;
    }
    else if(m==C+1){
        index_m = i * (H * W * C) + m * (H * W) + s * W + t;
        V[index_m] = 1;
    }
    else{
        index_m = i * (H * W * C) + 0 * (H * W) + s * W + t;
        V[index_m] = 1;
    }

  }
}


template <typename Dtype>
void OneToManyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* U = bottom[0]->gpu_data();
  Dtype* V = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), (Dtype)0, V);
  const int nthreads = num_ * height_ * width_;
  OneToManyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_, many_, height_, width_, U, V);
}

template <typename Dtype>
void OneToManyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  const Dtype* top_diff = top[0]->gpu_diff();

  const Dtype* top_data = top[0]->gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  Dtype* scale_data = scale_.mutable_gpu_data();  


  // Compute inner1d(top_diff, top_data)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num_ * num_pixels_),
      CAFFE_CUDA_NUM_THREADS>>>(num_, many_, num_pixels_,
      top_diff, top_data, scale_data);

  caffe_copy(num_ * num_pixels_, scale_data, bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(OneToManyLayer);


}  // namespace caffe
