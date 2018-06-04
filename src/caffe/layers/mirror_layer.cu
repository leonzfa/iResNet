#include <cfloat>
#include <vector>

#include "caffe/layers/mirror_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MirrorForward(const int nthreads, const int num,
                                    const int channels, const int height, const int width, const Dtype* U, Dtype* V) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int top_index, bottom_index;
    top_index    = (n * channels * height + c * height + h) * width + (width - 1 - w);
    bottom_index = (n * channels * height + c * height + h) * width + w;
    V[top_index] = U[bottom_index];
  }
}

template <typename Dtype>
__global__ void MirrorBackward(const int nthreads, const int num,
                                    const int channels, const int height, const int width, const Dtype* dV, Dtype* dU) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int top_index, bottom_index;
    top_index    = (n * channels * height + c * height + h) * width + (width - 1 - w);
    bottom_index = (n * channels * height + c * height + h) * width + w;
    dU[bottom_index] = dV[top_index];
  }
}

template <typename Dtype>
void MirrorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* U = bottom[0]->gpu_data();
    Dtype* V = top[0]->mutable_gpu_data();

    int num, channels, height, width;
    num = bottom[0]->num();
    channels = bottom[0]->channels(); // should be 1
    height   = bottom[0]->height();
    width    = bottom[0]->width();

    const int nthreads = bottom[0]->count();
    MirrorForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num, channels, height, width, U, V);
}


template <typename Dtype>
void MirrorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //std::cout<<"backward gpu"<<std::endl;
    if(propagate_down[0]){
        const Dtype* dV = top[0]->gpu_diff();
        Dtype* dU = bottom[0]->mutable_gpu_diff();

        int num, channels, height, width;
        num = bottom[0]->num();
        channels = bottom[0]->channels(); // should be 1
        height   = bottom[0]->height();
        width    = bottom[0]->width();

        const int nthreads = bottom[0]->count();
        MirrorBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
            CAFFE_CUDA_NUM_THREADS>>>(nthreads, num, channels, height, width, dV, dU);
    }
    else
        return;
}

INSTANTIATE_LAYER_GPU_FUNCS(MirrorLayer);

}  // namespace caffe
