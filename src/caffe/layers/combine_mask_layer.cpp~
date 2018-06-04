#include <cfloat>
#include <vector>

#include "caffe/layers/error_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ErrorLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  error_type_ = this->layer_param_.error_label_param().error_type();
  plateau_ = this->layer_param_.error_label_param().plateau();

  LayerParameter diff_param;
  diff_param.mutable_eltwise_param()->add_coeff(1.);
  diff_param.mutable_eltwise_param()->add_coeff(-1.);
  diff_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  diff_layer_.reset(new EltwiseLayer<Dtype>(diff_param));
  diff_layer_->SetUp(bottom, top);
}

template <typename Dtype>
void ErrorLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  diff_layer_->Reshape(bottom, top);
  mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());  
}

template <typename Dtype>
void ErrorLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void ErrorLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ErrorLabelLayer);
#endif

INSTANTIATE_CLASS(ErrorLabelLayer);
REGISTER_LAYER_CLASS(ErrorLabel);

}  // namespace caffe
