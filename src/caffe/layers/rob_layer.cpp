#include <cfloat>
#include <vector>

#include "caffe/layers/rob_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/downsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void ROBLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   num_dataset_ = static_cast<int>(bottom.size() / 3);
   CHECK(bottom.size() - num_dataset_ * 3 == 0) << "ROB Layer' bottom size is multiple of 3.";

   // Downsample height and width
   // In augmentation layer, the outputs have been cropped to 1280*256 (or bigger)
   // mode = 0, no downsample, but crop  
   // mode = 1, downsample 640*128(, and crop)
   // mode = 2, downsample to 320*64, and pad
  bottom_copy_.resize(bottom.size());
  for (int i = 0; i < bottom.size(); ++i) {
    bottom_copy_[i].reset(new Blob<Dtype>(bottom[i]->num(), bottom[i]->channels(), bottom[i]->width(), bottom[i]->height()));
    //caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(), bottom_copy_[i].get()->mutable_gpu_data()); // move to forward_gpu
  }


  int iC = bottom[0]->channels(); // image
  int dC = bottom[2]->channels(); // disp
  int N  = bottom[0]->num();

  height_half_ = static_cast<int>(bottom[0]->height() / 2);
  width_half_  = static_cast<int>(bottom[0]->width()  / 2);
  height_quarter_ = static_cast<int>(bottom[0]->height() / 4);
  width_quarter_  = static_cast<int>(bottom[0]->width()  / 4);
  target_height_ = this->layer_param_.rob_param().target_height();
  target_width_ = this->layer_param_.rob_param().target_width();   



   LayerParameter downsample_half_param;
   downsample_half_param.mutable_downsample_param()->set_top_height(height_half_);
   downsample_half_param.mutable_downsample_param()->set_top_width(width_half_);

   LayerParameter downsample_quarter_param;
   downsample_quarter_param.mutable_downsample_param()->set_top_height(height_quarter_);
   downsample_quarter_param.mutable_downsample_param()->set_top_width(width_quarter_);


  // blob
   downsampled_left_half_.resize(num_dataset_);
   downsampled_left_quarter_.resize(num_dataset_);
   downsampled_right_half_.resize(num_dataset_);
   downsampled_right_quarter_.resize(num_dataset_);
   downsampled_disp_half_.resize(num_dataset_);
   downsampled_disp_quarter_.resize(num_dataset_);

  for (int i = 0; i < num_dataset_; ++i) {
    // left
    downsampled_left_half_[i].reset(new Blob<Dtype>(N, iC, height_half_, width_half_));
    downsampled_left_quarter_[i].reset(new Blob<Dtype>(N, iC, height_quarter_, width_quarter_));
    // right
    downsampled_right_half_[i].reset(new Blob<Dtype>(N, iC, height_half_, width_half_));
    downsampled_right_quarter_[i].reset(new Blob<Dtype>(N, iC, height_quarter_, width_quarter_));
    // disp
    downsampled_disp_half_[i].reset(new Blob<Dtype>(N, dC, height_half_, width_half_));
    downsampled_disp_quarter_[i].reset(new Blob<Dtype>(N, dC, height_quarter_, width_quarter_));   
  }


  // vec
  downsample_left_half_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_right_half_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_disp_half_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_left_quarter_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_right_quarter_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_disp_quarter_bottom_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_left_half_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_right_half_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_disp_half_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_left_quarter_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_right_quarter_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];
  downsample_disp_quarter_top_vec_ = new vector<Blob<Dtype>*>[num_dataset_];

  // layer
  downsample_left_half_layer_.resize(num_dataset_);
  downsample_left_quarter_layer_.resize(num_dataset_);
  
  downsample_right_half_layer_.resize(num_dataset_);
  downsample_right_quarter_layer_.resize(num_dataset_);
  
  downsample_disp_half_layer_.resize(num_dataset_);
  downsample_disp_quarter_layer_.resize(num_dataset_);


   // For every dataset, instance three downsample layer, for left, right, disparity.
   for(int i = 0; i < num_dataset_; i++)
   {
     // for left image
     downsample_left_half_bottom_vec_[i].clear();
     downsample_left_half_bottom_vec_[i].push_back(bottom[i*3]);
     downsample_left_half_top_vec_[i].clear();
     downsample_left_half_top_vec_[i].push_back(downsampled_left_half_[i].get());     
     downsample_left_half_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_half_param));
     downsample_left_half_layer_[i]->SetUp(downsample_left_half_bottom_vec_[i], downsample_left_half_top_vec_[i]);

     downsample_left_quarter_bottom_vec_[i].clear();
     downsample_left_quarter_bottom_vec_[i].push_back(bottom[i*3]);
     downsample_left_quarter_top_vec_[i].clear();
     downsample_left_quarter_top_vec_[i].push_back(downsampled_left_quarter_[i].get());     
     downsample_left_quarter_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_quarter_param));
     downsample_left_quarter_layer_[i]->SetUp(downsample_left_quarter_bottom_vec_[i], downsample_left_quarter_top_vec_[i]);


     // for right image
     downsample_right_half_bottom_vec_[i].clear();
     downsample_right_half_bottom_vec_[i].push_back(bottom[i*3+1]);
     downsample_right_half_top_vec_[i].clear();
     downsample_right_half_top_vec_[i].push_back(downsampled_right_half_[i].get());     
     downsample_right_half_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_half_param));
     downsample_right_half_layer_[i]->SetUp(downsample_right_half_bottom_vec_[i], downsample_right_half_top_vec_[i]);

     downsample_right_quarter_bottom_vec_[i].clear();
     downsample_right_quarter_bottom_vec_[i].push_back(bottom[i*3+1]);
     downsample_right_quarter_top_vec_[i].clear();
     downsample_right_quarter_top_vec_[i].push_back(downsampled_right_quarter_[i].get());     
     downsample_right_quarter_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_quarter_param));
     downsample_right_quarter_layer_[i]->SetUp(downsample_right_quarter_bottom_vec_[i], downsample_right_quarter_top_vec_[i]);
     

     // for disparity
     downsample_disp_half_bottom_vec_[i].clear();
     downsample_disp_half_bottom_vec_[i].push_back(bottom[i*3+2]);
     downsample_disp_half_top_vec_[i].clear();
     downsample_disp_half_top_vec_[i].push_back(downsampled_disp_half_[i].get());     
     downsample_disp_half_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_half_param));
     downsample_disp_half_layer_[i]->SetUp(downsample_disp_half_bottom_vec_[i], downsample_disp_half_top_vec_[i]);

     downsample_disp_quarter_bottom_vec_[i].clear();
     downsample_disp_quarter_bottom_vec_[i].push_back(bottom[i*3+2]);
     downsample_disp_quarter_top_vec_[i].clear();
     downsample_disp_quarter_top_vec_[i].push_back(downsampled_disp_quarter_[i].get());     
     downsample_disp_quarter_layer_[i].reset(new DownsampleLayer<Dtype>(downsample_quarter_param));
     downsample_disp_quarter_layer_[i]->SetUp(downsample_disp_quarter_bottom_vec_[i], downsample_disp_quarter_top_vec_[i]);
   }


  coeffs_ = vector<Dtype>(num_dataset_ + 1, 1);
  coeffs_[0] = 0;
  coeffs_[num_dataset_] = 1;
  if (this->layer_param().rob_param().coeff_size()) {
    for (int i = 1; i < num_dataset_ + 1; ++i) {
      coeffs_[i] = this->layer_param().rob_param().coeff(i-1);
    }
  }
  LOG(INFO) << ("8");
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), target_height_, target_width_); // left
  top[1]->Reshape(bottom[1]->num(), bottom[1]->channels(), target_height_, target_width_); // right
  top[2]->Reshape(bottom[2]->num(), bottom[2]->channels(), target_height_, target_width_); // disparity

  offsets = vector<int>(2, 0);
  
  //rand_height_.Reshape(1,1,1,1);
  //rand_width_.Reshape(1,1,1,1);

  rand_vec_.Reshape(2,1,1,1);
  seg_vec_.Reshape(1,1,1,1);
  mode_vec_.Reshape(1,1,1,1);
}

template <typename Dtype>
void ROBLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  return;
}

template <typename Dtype>
void ROBLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  return;
}

template <typename Dtype>
void ROBLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(ROBLayer);
#endif

INSTANTIATE_CLASS(ROBLayer);
REGISTER_LAYER_CLASS(ROB);

}  // namespace caffe
