#include <cfloat>
#include <vector>

#include "caffe/layers/shift_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	template <typename Dtype>
	__global__ void ShiftForward(const int nthreads, int N, int C,
			int H, int W, const Dtype* U, Dtype* V, const int direction, const int displacement) {

		CUDA_KERNEL_LOOP(index, nthreads) {

			const int t = index % W;
			const int s = (index / W) % H;
			const int j = (index / (W * H)) % C;
			const int i = index / (W * H * C);
			
			int bt = t;
			int bs = s;

			if(direction == 0){
				bt = t + displacement; // displacement > 0
                V[i * (C * H * W) + j * (H * W) + W * s + t] = (bt >= 0 & bt < W) ? U[i * (C * H * W) + j * (H * W) + W * s + bt] : 0;
			}
			else{
				bs = s + displacement; // displacement > 0
                V[i * (C * H * W) + j * (H * W) + W * s + t] = (bs >= 0 & bs < H) ? U[i * (C * H * W) + j * (H * W) + W * bs + t] : 0;
			}

	  }
	}





template <typename Dtype>
void ShiftLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const int direction = this->layer_param().shift_param().direction();
  const int displacement = this->layer_param().shift_param().displacement(); 
  ShiftForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), direction, displacement);
}

template <typename Dtype>
__global__ void ShiftBackward(const int nthreads, int N, int C,
		int H, int W, const Dtype* U, Dtype* V, const int direction, const int displacement) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % W;
		const int s = (index / W) % H;
		const int j = (index / (W * H)) % C;
		const int i = index / (W * H * C);
		
		int bt = t;
		int bs = s;
        // U: top_diff
		// V: bottom_diff
		if(direction == 0){
			bt = t - displacement; // displacement > 0
            V[i * (C * H * W) + j * (H * W) + W * s + t] = (bt >= 0 & bt < W) ? U[i * (C * H * W) + j * (H * W) + W * s + bt] : 0;
		}
		else{
			bs = s - displacement; // displacement > 0
            V[i * (C * H * W) + j * (H * W) + W * s + t] = (bs >= 0 & bs < H) ? U[i * (C * H * W) + j * (H * W) + W * bs + t] : 0;
		}

  }
}

template <typename Dtype>
void ShiftLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
	    const int count = top[0]->count();
	    const int direction = this->layer_param().shift_param().direction();
	    const int displacement = this->layer_param().shift_param().displacement(); 
	    ShiftBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	        count, bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), direction, displacement);
}

INSTANTIATE_LAYER_GPU_FUNCS(ShiftLayer);

}  // namespace caffe
