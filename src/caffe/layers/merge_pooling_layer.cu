#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/merge_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MergePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    const Dtype* const bottom_slice =
        bottom_data + (n * channels * 4 + c * 4) * pooled_height * pooled_width;
    top_data[(n * channels + c) * height * width + hstart * width + wstart]           =  bottom_slice[0 * pooled_height * pooled_width + ph * pooled_width + pw];
    top_data[(n * channels + c) * height * width + hstart * width + wstart + 1]       =  bottom_slice[1 * pooled_height * pooled_width + ph * pooled_width + pw];
    top_data[(n * channels + c) * height * width + (hstart + 1) * width + wstart]     =  bottom_slice[2 * pooled_height * pooled_width + ph * pooled_width + pw];
    top_data[(n * channels + c) * height * width + (hstart + 1) * width + wstart + 1] =  bottom_slice[3 * pooled_height * pooled_width + ph * pooled_width + pw];
  }
}

template <typename Dtype>
void MergePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = static_cast<int>(ceil(static_cast<float>(top[0]->count() / 4)));  
  MergePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);  
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MergePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * height * width;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    bottom_diff[(n * channels * 4 + c * 4 + 0) * pooled_height * pooled_width + ph * pooled_width + pw] = top_diff_slice[hstart * width + wstart];
    bottom_diff[(n * channels * 4 + c * 4 + 1) * pooled_height * pooled_width + ph * pooled_width + pw] = top_diff_slice[hstart * width + wstart + 1];
    bottom_diff[(n * channels * 4 + c * 4 + 2) * pooled_height * pooled_width + ph * pooled_width + pw] = top_diff_slice[(hstart + 1) * width + wstart];
    bottom_diff[(n * channels * 4 + c * 4 + 3) * pooled_height * pooled_width + ph * pooled_width + pw] = top_diff_slice[(hstart + 1) * width + wstart + 1];
  }
}

template <typename Dtype>
void MergePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();  
  const int count = static_cast<int>(ceil(static_cast<float>(bottom[0]->count() / 4)));
  caffe_gpu_set(count, Dtype(0.), bottom_diff); 
  MergePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);  
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(MergePoolingLayer);

}  // namespace caffe
