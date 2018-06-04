#include <cfloat>
#include <vector>

#include "caffe/layers/random_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <iostream>
#include <ctime>
#include <math.h>

using namespace std;

namespace caffe {


template <typename Dtype>
__global__ void crop_copy_gpu(const int nthreads, int N, int C, int bH, int bW, int tH, int tW,
		const Dtype* bottom_data, Dtype* top_data, const int offsets_w, const int offsets_h, const Dtype nan) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % tW; // W index
		const int s = (index / tW) % tH; // H index		
                const int j = (index / (tW * tH)) % C; // C index
		const int i = index / (tW * tH * C); // N index
                int nt = t + offsets_w;
                int ns = s + offsets_h;
		int b_index = i * (C * bH * bW) + j * (bH * bW) + bW * ns + nt;
                Dtype pad_value = 128;

                if(j==C-1) // last channel
                  pad_value = nan;


                if((0<= nt & nt < bW) && (0<= ns & ns < bH))
	            top_data[index] = bottom_data[b_index];
                else
		    top_data[index] = pad_value;

  }
}


template <typename Dtype>
void RandomCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


  std::srand ((unsigned)time(NULL));
  
  // Random crop  
  const int N = bottom[0]->num();
  const int C = bottom[0]->channels(); // image
  const int bH = bottom[0]->height();
  const int bW = bottom[0]->width();

  const int tH = top[0]->height();
  const int tW = top[0]->width();
  const int count = top[0]->count();



  if (this->phase_ == TRAIN) {
    // Random offsets
    offsets[0] = int(caffe_rng_rand());
    offsets[1] = int(caffe_rng_rand());
 
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
    }
    else{
      offsets[1] = static_cast<int>((bH - tH)/2);   
    }  
  } 
  else {    
    offsets[0] = static_cast<int>((bW - tW)/2);
    offsets[1] = static_cast<int>((bH - tH)/2);       
  }



  crop_copy_gpu<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, N, C, bH, bW, tH, tW, bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), offsets[0], offsets[1], std::numeric_limits<Dtype>::signaling_NaN());

}


template <typename Dtype>
void RandomCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomCropLayer);

}  // namespace caffe
