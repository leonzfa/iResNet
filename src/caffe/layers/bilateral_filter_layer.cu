#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bilateral_filter_layer.hpp"



namespace caffe {


template <typename Dtype>
__global__ void CalculateDiffGPU(const int nthreads, int N, int C, int H, int W,
		const Dtype* bottom_data, Dtype* top_data, const int direction) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % W; // W index
		const int s = (index / W) % H; // H index		
                const int j = (index / (W * H)) % C; // C index
		const int i = index / (W * H * C); // N index

		int nt = t;
		int ns = s;


		if(direction == 0){		nt = t - 1;} 		// 0: left
		else if(direction == 1){	nt = t + 1;}		// 1: right
		else if(direction == 2){	ns = s - 1;}		// 2: up
		else{				ns = s + 1;}		// 3: bottom

		int nindex = i * (C * H * W) + j * (H * W) + W * ns + nt;

		if(nt <  0 | nt >= W | ns < 0 | ns >= H){ // out of range
			top_data[index] = 0;}
		else{
			top_data[index] = bottom_data[index] - bottom_data[nindex];
		}
  }
}



template <typename Dtype>
__global__ void ChannelMeanForwardGPU(const int nthreads, int N, int C, int H, int W,
		const Dtype* bottom_data, Dtype* top_data, const Dtype scaling) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % W;
		const int s = (index / W) % H;		
		const int i = index / (W * H);
		Dtype tmp = 0;

		for( int j = 0; j < C; j++){
			tmp += bottom_data[i * (C * H * W) + j * (H * W) + W * s + t];
		}
		top_data[index] = tmp / C * scaling; // scaling = -1/sigma
  }
}


template <typename Dtype>
void BilateralFilterLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();
	const int C = bottom[0]->channels();
	const int N = bottom[0]->num();  
	const int count = bottom[0]->count();
	const int weight_count = N * H * W;

	const Dtype* similarity_data;
	Dtype* dl_data = diff_left_.mutable_gpu_data();
	Dtype* dr_data = diff_right_.mutable_gpu_data();
	Dtype* du_data = diff_up_.mutable_gpu_data();
	Dtype* db_data = diff_bottom_.mutable_gpu_data();

	if(bottom.size() == 1){
		similarity_data = bottom[0]->gpu_data();}
	else{
		similarity_data = bottom[1]->gpu_data();}

	// calculate diff along four directions
	CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
	      CAFFE_CUDA_NUM_THREADS>>>(count, N, C, H, W, similarity_data, dl_data, 0);

	CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
	      CAFFE_CUDA_NUM_THREADS>>>(count, N, C, H, W, similarity_data, dr_data, 1);

	CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
	      CAFFE_CUDA_NUM_THREADS>>>(count, N, C, H, W, similarity_data, du_data, 2);

	CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
	      CAFFE_CUDA_NUM_THREADS>>>(count, N, C, H, W, similarity_data, db_data, 3);



	Dtype sigma = this->layer_param_.bilateral_filter_param().sigma();
	Dtype scaling = -1 / sigma;

	// sum along channel
	// -- left
	caffe_gpu_abs(bottom[0]->count(), dl_data, dl_abs_.mutable_gpu_data());
	ChannelMeanForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, dl_abs_.gpu_data(), dl_abs_sum_.mutable_gpu_data(), scaling);
	caffe_gpu_exp(count, dl_abs_sum_.gpu_data(), dl_abs_sum_.mutable_gpu_data());
	// -- right
	caffe_gpu_abs(bottom[0]->count(), dr_data, dr_abs_.mutable_gpu_data());
	ChannelMeanForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, dr_abs_.gpu_data(), dr_abs_sum_.mutable_gpu_data(), scaling);
	caffe_gpu_exp(count, dr_abs_sum_.gpu_data(), dr_abs_sum_.mutable_gpu_data());
	// -- up
	caffe_gpu_abs(bottom[0]->count(), du_data, du_abs_.mutable_gpu_data());
	ChannelMeanForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, du_abs_.gpu_data(), du_abs_sum_.mutable_gpu_data(), scaling);
	caffe_gpu_exp(count, du_abs_sum_.gpu_data(), du_abs_sum_.mutable_gpu_data());
	// -- bottom
	caffe_gpu_abs(bottom[0]->count(), db_data, db_abs_.mutable_gpu_data());
	ChannelMeanForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, db_abs_.gpu_data(), db_abs_sum_.mutable_gpu_data(), scaling);
	caffe_gpu_exp(count, db_abs_sum_.gpu_data(), db_abs_sum_.mutable_gpu_data());
	
	//== similarity along four direction have been calculated.

	// iterative propagation
	// horizontal
	for (int i = 0; i < num_iterations_; ++i) {
		bilateral_iterations_left_[i]->Forward_gpu(bilateral_iterations_left_bottom_vec_[i], bilateral_iterations_left_top_vec_[i]);
		bilateral_iterations_right_[i]->Forward_gpu(bilateral_iterations_right_bottom_vec_[i], bilateral_iterations_right_top_vec_[i]);
	}
	horizontal_layer_->Forward(horizontal_bottom_vec_, horizontal_top_vec_); // left + right - input

	// vertical
        for (int i = 0; i < num_iterations_; ++i) {
		bilateral_iterations_up_[i]->Forward_gpu(bilateral_iterations_up_bottom_vec_[i], bilateral_iterations_up_top_vec_[i]);
		bilateral_iterations_bottom_[i]->Forward_gpu(bilateral_iterations_bottom_bottom_vec_[i], bilateral_iterations_bottom_top_vec_[i]);
		
	}
	//update
	vertical_layer_->Forward(vertical_bottom_vec_, vertical_top_vec_); // up + bottom - input

}





// D_data: N1HW = exp(-1/sigma/C * sum( abs(d)))
//


template <typename Dtype>
__global__ void ChannelMeanBackwardGPU(const int nthreads, int N, int C, int H, int W,
		const Dtype* D_data, const Dtype* d_sign, Dtype* output_data, const Dtype scaling, const int direction) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int t = index % W;
		const int s = (index / W) % H;		
		const int i = index / (W * H);
	
		int index_shift;
		int index0;
		int index0_shift;

		switch(direction){
			case 0: index_shift = i * ( H * W)  + W * s + (t+1); break;  //left
			case 1: index_shift = i * ( H * W)  + W * s + (t-1); break;  //right
			case 2: index_shift = i * ( H * W)  + W * (s+1) + t; break;  //up
			default: index_shift = i * ( H * W)  + W * (s-1) + t; break; //bottom
		}
		for(int j = 0; j < C; j++){
			index0 = i * (C * H * W) + j * (H * W) + W * s + t;
			switch(direction){
				case 0: index0_shift = i * ( H * W) + j * (H * W) + W * s + (t+1); break;  //left
				case 1: index0_shift = i * ( H * W) + j * (H * W) + W * s + (t-1); break;  //right
				case 2: index0_shift = i * ( H * W) + j * (H * W) + W * (s+1) + t; break;  //up
				default: index0_shift = i * ( H * W) + j * (H * W) + W * (s-1) + t; break; //bottom
			}
			if((direction == 0 & t+1>=W) | (direction == 1 & t-1<0) | (direction == 2 & s+1>=H) | (direction == 3 & s-1<0)){
				output_data[index0] = 0;}
			else{
				output_data[index0] = D_data[index] * d_sign[index0] - D_data[index_shift] * d_sign[index0_shift];
				output_data[index0] = output_data[index0] * scaling / C;  // scaling = -1/sigma
			}
		}
  }
}


template <typename Dtype>
void BilateralFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	vector<bool> prop_down(1,true);
	

	vertical_layer_->Backward(vertical_top_vec_, prop_down, vertical_bottom_vec_); // up + bottom - input	
        for (int i = 0; i < num_iterations_; ++i) {
		bilateral_iterations_bottom_[i]->Backward_gpu(bilateral_iterations_bottom_top_vec_[i], prop_down, bilateral_iterations_bottom_bottom_vec_[i]);
		bilateral_iterations_up_[i]->Backward_gpu(bilateral_iterations_up_top_vec_[i], prop_down, bilateral_iterations_up_bottom_vec_[i]);
	}
	horizontal_layer_->Backward(horizontal_top_vec_, prop_down, horizontal_bottom_vec_); // left + right - input
	for (int i = 0; i < num_iterations_; ++i) {
		bilateral_iterations_right_[i]->Backward_gpu(bilateral_iterations_right_top_vec_[i], prop_down, bilateral_iterations_right_bottom_vec_[i]);
		bilateral_iterations_left_[i]->Backward_gpu(bilateral_iterations_up_top_vec_[i], prop_down, bilateral_iterations_up_bottom_vec_[i]);
	}

	// 
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();
	const int C = bottom[0]->channels();
	const int N = bottom[0]->num();  
	const int count = bottom[0]->count();
	const int weight_count = N * H * W;

	const Dtype* dl_data = diff_left_.gpu_data();
	const Dtype* dr_data = diff_right_.gpu_data();
	const Dtype* du_data = diff_up_.gpu_data();
	const Dtype* db_data = diff_bottom_.gpu_data();

	caffe_gpu_sign(count, dl_data, dl_data_sign_.mutable_gpu_data());
	caffe_gpu_sign(count, dr_data, dr_data_sign_.mutable_gpu_data());
	caffe_gpu_sign(count, du_data, du_data_sign_.mutable_gpu_data());
	caffe_gpu_sign(count, db_data, db_data_sign_.mutable_gpu_data());

	Dtype sigma = this->layer_param_.bilateral_filter_param().sigma();
	Dtype scaling = -1 / sigma;

	ChannelMeanBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, dl_abs_sum_.gpu_data(), dl_data_sign_.gpu_data(), local_left_diff_.mutable_gpu_data(), scaling, 0);
	ChannelMeanBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, dr_abs_sum_.gpu_data(), dr_data_sign_.gpu_data(), local_right_diff_.mutable_gpu_data(), scaling, 1);
	ChannelMeanBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, du_abs_sum_.gpu_data(), du_data_sign_.gpu_data(), local_up_diff_.mutable_gpu_data(), scaling, 2);
	ChannelMeanBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(weight_count),
	      CAFFE_CUDA_NUM_THREADS>>>(weight_count, N, C, H, W, db_abs_sum_.gpu_data(), db_data_sign_.gpu_data(), local_bottom_diff_.mutable_gpu_data(), scaling, 3);
	//
        caffe_gpu_mul(count, local_left_diff_.mutable_gpu_data(),   dl_abs_sum_.gpu_diff(), local_left_diff_.mutable_gpu_data());
        caffe_gpu_mul(count, local_right_diff_.mutable_gpu_data(),  dr_abs_sum_.gpu_diff(), local_right_diff_.mutable_gpu_data());
        caffe_gpu_mul(count, local_up_diff_.mutable_gpu_data(),     du_abs_sum_.gpu_diff(), local_up_diff_.mutable_gpu_data());
        caffe_gpu_mul(count, local_bottom_diff_.mutable_gpu_data(), db_abs_sum_.gpu_diff(), local_bottom_diff_.mutable_gpu_data());
  
	Dtype* propagated_diff;
	if(bottom.size() == 1){
		propagated_diff = bottom[0]->mutable_gpu_diff();}
	else{
		propagated_diff = bottom[1]->mutable_gpu_diff();}

	caffe_gpu_axpy(count, Dtype(1.), local_left_diff_.gpu_data(), propagated_diff);
	caffe_gpu_axpy(count, Dtype(1.), local_right_diff_.gpu_data(), propagated_diff);
	caffe_gpu_axpy(count, Dtype(1.), local_up_diff_.gpu_data(), propagated_diff);
	caffe_gpu_axpy(count, Dtype(1.), local_bottom_diff_.gpu_data(), propagated_diff);


}

INSTANTIATE_LAYER_GPU_FUNCS(BilateralFilterLayer);

}	// namespace caffe
