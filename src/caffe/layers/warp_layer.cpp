#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/warp_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
void WarpDisparityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tWarp Disparity Layer:: LayerSetUp: \t";
        to_compute_dU_ = true;
	output_H_ = bottom[0]->shape(2);
	output_W_ = bottom[0]->shape(3);
	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Both bottom blobs must have same width";
        CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Both bottom blobs must have same height";
        //CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Both bottom blobs must have same number of channels";
	
	vector<int> shape_output(2);
	shape_output[0] = output_H_ * output_W_; shape_output[1] = 2;
	output_grid.Reshape(shape_output);

        
        //disparity_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
        //Dtype* data = disparity_grid.mutable_cpu_data();
  
	for(int i=0; i<output_H_ * output_W_; ++i) {
		data[2 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
		data[2 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
	}

	// initialize the matrix for input grid
	std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(3);
	shape_input[0] = bottom[1]->shape(0); shape_input[1] = output_H_ * output_W_; shape_input[2] = 2;
	input_grid.Reshape(shape_input);

	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void WarpDisparityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tWarp Disparity Layer:: Reshape: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);

	// reshape dTheta_tmp
	// vector<int> dTheta_tmp_shape(4);

	// dTheta_tmp_shape[0] = N;
	// dTheta_tmp_shape[1] = 2;
	// dTheta_tmp_shape[2] = 3;
	// dTheta_tmp_shape[3] = output_H_ * output_W_ * C;

	// dTheta_tmp.Reshape(dTheta_tmp_shape);

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] = output_H_ * output_W_ * C;
	all_ones_2.Reshape(all_ones_2_shape);

	// reshape full_theta
	vector<int> full_theta_shape(2);
	full_theta_shape[0] = N;
	full_theta_shape[1] = 6;
	full_theta.Reshape(full_theta_shape);

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
Dtype WarpDisparityLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {

	bool debug = false;

	string prefix = "\t\tSpatial Transformer Layer:: transform_forward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
	if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

	Dtype res = (Dtype)0.;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;

	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

        m = int(floor(x)); n = int(floor(y)); w = 0;
	if(debug) std::cout<<prefix<<"1: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

        m = int(floor(x) + 1); n = int(floor(y)); w = 0;
	if(debug) std::cout<<prefix<<"2: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

        m = int(floor(x)); n = int(floor(y) + 1); w = 0;
	if(debug) std::cout<<prefix<<"3: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

        m = int(floor(x) + 1); n = int(floor(y) + 1); w = 0;
	if(debug) std::cout<<prefix<<"4: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

	return res;
}

template <typename Dtype>
void WarpDisparityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tSpatial Transformer Layer:: Forward_cpu: \t";
	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* D = bottom[1]->cpu_data();
	const Dtype* output_grid_data = output_grid.cpu_data();

	Dtype* input_grid_data = input_grid.mutable_cpu_data();
	Dtype* V = top[0]->mutable_cpu_data();

	caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_set(top[0]->count(), (Dtype)0, V);

	// for each input
	for(int i = 0; i < N; ++i) {
                const Dtype* disparity   = D + (output_H_ * output_W_) * i;
                const Dtype* in  = U + (output_H_ * output_W_ * C) * i;
                Dtype* out = V + (output_H_ * output_W_ * C) * i;

		int row_idx; 
                Dtype px, py;

		for(int j = 0; j < C; ++j)
                        for(int s = 0; s < H; ++s)
                                for(int t = 0; t < W; ++t) {
                                        row_idx = W * s + t;
                                        px = output_grid_data[row_idx * 2];
                                        py = output_grid_data[row_idx * 2 + 1] + disparity[row_idx] / output_W_ * 2;
                                        out[j * H * W + row_idx] = transform_forward_cpu(
                                                        in + j * H * W,  px, py);
				}
	}

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void WarpDisparityLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
		const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {

	bool debug = false;

	string prefix = "\t\tSpatial Transformer Layer:: transform_backward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

	Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;
	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	if(debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void WarpDisparityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		string prefix = "\t\tSpatial Transformer Layer:: Backward_cpu: \t";

                //CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
                //		" CHECK in st_layer.cpp file. Line number: 420-421." << std::endl;

		if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		const Dtype* input_grid_data = input_grid.cpu_data();
		const Dtype* U = bottom[0]->cpu_data();

		Dtype* dU = bottom[0]->mutable_cpu_diff();
		// Dtype* dTheta = bottom[1]->mutable_cpu_diff();
                Dtype* dD = bottom[1]->mutable_cpu_diff();
		Dtype* input_grid_diff = input_grid.mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		//caffe_set(bottom[1]->count(), (Dtype)0, dTheta);
                caffe_set(bottom[1]->count(), (Dtype)0, dD);
		caffe_set(input_grid.count(), (Dtype)0, input_grid_diff);

		for(int i = 0; i < N; ++i) {

			const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
			Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;
                        Dtype* disparity_diff = dD + (output_H_ * output_W_) * i;

			int row_idx; Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;

			for(int s = 0; s < output_H_; ++s)
				for(int t = 0; t < output_W_; ++t) {

					row_idx = output_W_ * s + t;

					px = coordinates[row_idx * 2];
					py = coordinates[row_idx * 2 + 1];

					for(int j = 0; j < C; ++j) {

						delta_dpx = delta_dpy = (Dtype)0.;

						transform_backward_cpu(dV[top[0]->offset(i, j, s, t)], U + bottom[0]->offset(i, j, 0, 0),
								px, py, dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);

						coordinates_diff[row_idx * 2] += delta_dpx;
						coordinates_diff[row_idx * 2 + 1] += delta_dpy;
					}

					dpx = coordinates_diff[row_idx * 2];
					dpy = coordinates_diff[row_idx * 2 + 1];

                                        disparity_diff[s * output_W_ + t] = dpy;


					// dTheta[6 * i] += dpx * (s * 1.0 / output_H_ * 2 - 1);
					// dTheta[6 * i + 1] += dpx * (t * 1.0 / output_W_ * 2 - 1);
					// dTheta[6 * i + 2] += dpx;
					// dTheta[6 * i + 3] += dpy * (s * 1.0 / output_H_ * 2 - 1);
					// dTheta[6 * i + 4] += dpy * (t * 1.0 / output_W_ * 2 - 1);
					// dTheta[6 * i + 5] += dpy;
				}
		}

		if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(WarpDisparityLayer);
#endif

INSTANTIATE_CLASS(WarpDisparityLayer);
REGISTER_LAYER_CLASS(WarpDisparity);

}  // namespace caffe
