#include <cfloat>
#include <vector>

#include "caffe/layers/rob_layer.hpp"
#include "caffe/layers/downsample_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <iostream>
#include <ctime>
#include <math.h>

using namespace std;

// inputs: 320 * 768
// mode 0: 320 * 768  --> 128 * 384
// mode 1: 160 * 384  --> 128 * 384
// mode 2: 80  * 192  --> 128 * 384
// output: 128 * 384

namespace caffe {


template <typename Dtype>
__global__ void crop_copy_gpu(const int nthreads, int N, int C, int bH, int bW, int tH, int tW,
		const Dtype* bottom_data, Dtype* top_data, const int offsets_w, const int offsets_h, const Dtype pad_value) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % tW; // W index
		const int s = (index / tW) % tH; // H index		
                const int j = (index / (tW * tH)) % C; // C index
		const int i = index / (tW * tH * C); // N index
                int nt = t + offsets_w;
                int ns = s + offsets_h;
		int b_index = i * (C * bH * bW) + j * (bH * bW) + bW * ns + nt;
                
                if((0<= nt & nt < bW) && (0<= ns & ns < bH))
	            top_data[index] = bottom_data[b_index];
                    //top_data[index] = Dtype(0);
                else
		    top_data[index] = pad_value;

  }
}


template <typename Dtype>
void ROBLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for (int i = 0; i < bottom.size(); ++i) {    
    caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(), bottom_copy_[i].get()->mutable_gpu_data());
  }

  std::srand ((unsigned)time(NULL));
  // Random select segmentation point
  Dtype seg = (static_cast<Dtype>(caffe_rng_rand()) / RAND_MAX);
  seg = seg - int(seg);
  cout<<"seg = "<<seg<<endl;

  int dataset = num_dataset_ - 1;
  for(int i=0; i < num_dataset_; i++){
    if(coeffs_[i] <= seg && seg < coeffs_[i+1]){
      dataset = i;
      break;
    }
  }

  // Random select operation mode
  Dtype select_mode = static_cast<Dtype>(caffe_rng_rand()) / RAND_MAX;
  select_mode = select_mode - int(select_mode);
  cout<<"select_mode = "<<select_mode<<endl;

  int mode = 2; 
  for(int i=0; i < 3; i++){
    if(i/Dtype(3.0) <= select_mode && select_mode < (i+1)/Dtype(3.0)){
      mode = i;
      break;
    }
  }




  // In our implementation, there are 4 datasets, i.e., SceneFlow, KITTI, MiddleBury, eth3d.
  // There 3 modes.
  //   in mode 0, we crop image in the original resolution, and output image whose resolution is 
  //   in mode 1, we first downsample the images to 1/2, then crop. The output images' resolution is 
  //   in mode 2, we first downsample the images to 1/4, then crop. The output images' resolution is 

  // Downsample

  //  mode = 0;
  cout<<"Choosing dataset = "<<dataset<<endl;
  cout<<"Operation mode = "<<mode<<endl;

  downsample_left_half_layer_[dataset]->Forward(downsample_left_half_bottom_vec_[dataset], downsample_left_half_top_vec_[dataset]);
  downsample_right_half_layer_[dataset]->Forward(downsample_right_half_bottom_vec_[dataset], downsample_right_half_top_vec_[dataset]);
  downsample_disp_half_layer_[dataset]->Forward(downsample_disp_half_bottom_vec_[dataset], downsample_disp_half_top_vec_[dataset]);

  downsample_left_quarter_layer_[dataset]->Forward(downsample_left_quarter_bottom_vec_[dataset], downsample_left_quarter_top_vec_[dataset]);
  downsample_right_quarter_layer_[dataset]->Forward(downsample_right_quarter_bottom_vec_[dataset], downsample_right_quarter_top_vec_[dataset]);
  downsample_disp_quarter_layer_[dataset]->Forward(downsample_disp_quarter_bottom_vec_[dataset], downsample_disp_quarter_top_vec_[dataset]);

  const Dtype*  tmp_left_data;
  const Dtype*  tmp_right_data;
  const Dtype*  tmp_disp_data;

  if(mode == 1){
      cout<<"half!!"<<endl;
      tmp_left_data  = downsampled_left_half_[dataset].get()->gpu_data();
      tmp_right_data = downsampled_right_half_[dataset].get()->gpu_data();
      tmp_disp_data  = downsampled_disp_half_[dataset].get()->gpu_data();
  }
  if(mode == 2){
      cout<<"quarter!!"<<endl;
      tmp_left_data  = downsampled_left_quarter_[dataset].get()->gpu_data();
      tmp_right_data = downsampled_right_quarter_[dataset].get()->gpu_data();
      tmp_disp_data  = downsampled_disp_quarter_[dataset].get()->gpu_data();
  }
  if(mode == 0){
      tmp_left_data  = bottom[dataset*3]->gpu_data();
      tmp_right_data = bottom[dataset*3+1]->gpu_data();
      tmp_disp_data  = bottom[dataset*3+2]->gpu_data();
  }
  LOG(INFO) << ("crop");
  // Random crop  

  int bW = bottom[0]->width();
  int bH = bottom[0]->height();
  if(mode == 1){
    bW = width_half_;
    bH = height_half_;}

  if(mode == 2){
    bW = width_quarter_;
    bH = height_quarter_;
  }



  const int tW = target_width_;
  const int tH = target_height_;

  const int iC = bottom[0]->channels(); // image
  const int dC = bottom[2]->channels(); // disp
  const int N  = bottom[0]->num();

  const int image_count = top[0]->count();
  const int disp_count  = top[2]->count();

  // Random offsets
  offsets[0] = int(caffe_rng_rand());
  offsets[1] = int(caffe_rng_rand());
//  offsets[1] = static_cast<int>(abs(caffe_rng_rand()));
  cout<<"random offsets = ("<<offsets[0]<<","<<offsets[1]<<")"<<endl;
  if(bW > tW){
    offsets[0] = offsets[0] % (bW - tW);
    offsets[0] = abs(offsets[0]);
  }else if( bW == tW){
    offsets[0] = 0;
  }else{
    offsets[0] = static_cast<int>((bW - tW)/2);   
  }
  
  if(bH > tH){
    offsets[1] = offsets[1] % (bH - tH);
    offsets[1] = abs(offsets[1]);
  }else if( bH == tH){
    offsets[1] = 0;
  }else{
    offsets[1] = static_cast<int>((bH - tH)/2);   
  }  

  cout<<"dC, bH, bW, tH, tW = (" <<dC<<", "<<bH<<", "<<bW<<", "<<tH<<", "<<tW<<")"<<endl;
  cout<<"final offsets = ("<<offsets[0]<<","<<offsets[1]<<")"<<endl;


  caffe_gpu_set(image_count, (Dtype)0., top[0]->mutable_gpu_data());
  caffe_gpu_set(image_count, (Dtype)0., top[1]->mutable_gpu_data());
  caffe_gpu_set(disp_count, (Dtype)0., top[2]->mutable_gpu_data());

  //LOG(INFO) << "Bottom shape: " << downsampled_left_half_[dataset].get()->shape_string();
  //LOG(INFO) << "Top shape: " << top[0]->shape_string();

  crop_copy_gpu<Dtype><<<CAFFE_GET_BLOCKS(image_count),
        CAFFE_CUDA_NUM_THREADS>>>(image_count, N, iC, bH, bW, tH, tW, tmp_left_data, top[0]->mutable_gpu_data(), offsets[0], offsets[1], Dtype(0.));


  crop_copy_gpu<Dtype><<<CAFFE_GET_BLOCKS(image_count),
        CAFFE_CUDA_NUM_THREADS>>>(image_count, N, iC, bH, bW, tH, tW, tmp_right_data, top[1]->mutable_gpu_data(), offsets[0], offsets[1], Dtype(0.));


  crop_copy_gpu<Dtype><<<CAFFE_GET_BLOCKS(disp_count),
        CAFFE_CUDA_NUM_THREADS>>>(disp_count, N, dC, bH, bW, tH, tW, tmp_disp_data, top[2]->mutable_gpu_data(), offsets[0], offsets[1], std::numeric_limits<Dtype>::signaling_NaN());
  //LOG(INFO) << ("Done.");
}


template <typename Dtype>
void ROBLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROBLayer);

}  // namespace caffe
