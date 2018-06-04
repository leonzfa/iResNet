#include <cfloat>
#include <vector>

#include "caffe/layers/shift_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ShiftLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    return;
}

template <typename Dtype>
void ShiftLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ShiftLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    return;
}

template <typename Dtype>
void ShiftLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
}

#ifdef CPU_ONLY
STUB_GPU(ShiftLayer);
#endif

INSTANTIATE_CLASS(ShiftLayer);
REGISTER_LAYER_CLASS(Shift);

}  // namespace caffe
