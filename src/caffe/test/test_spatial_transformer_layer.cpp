#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/custom_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

    // Reference affine transformer for checking results
    // compute coordinates and sample feature map explicitly using loops
    template <typename Dtype>
    void affine_transform(const Blob<Dtype>* in, const Blob<Dtype>* theta, Blob<Dtype>* out) {
        int num = in->shape(0);
        int channels = in->shape(1);
        int height = in->shape(2);
        int width = in->shape(3);
        Dtype* out_data = out->mutable_cpu_data();
        caffe_set<Dtype>(out->count(), 0, out_data);
        const Dtype* theta_data = theta->cpu_data();
        for (int n = 0; n < num; ++n) {
            for (int h = 0; h < height; ++h) {
                Dtype ty = h / (Dtype) (height - 1) * (Dtype) 2. - (Dtype) 1.;
                for (int w = 0; w < width; ++w) {
                    Dtype tx = w / (Dtype) (width - 1)*(Dtype) 2. - (Dtype) 1.;
                    Dtype sx = tx * theta_data[n * 6] + ty * theta_data[n * 6 + 1] + theta_data[n * 6 + 2];
                    Dtype sy = tx * theta_data[n * 6 + 3] + ty * theta_data[n * 6 + 4] + theta_data[n * 6 + 5];
                    sx = (sx + 1.) / (Dtype) 2. * (width - 1);
                    sy = (sy + 1.) / (Dtype) 2. * (height - 1);
                    for (int c = 0; c < channels; ++c) {
                        for (int hh = 0; hh < height; ++hh) {
                            for (int ww = 0; ww < width; ++ww) {
                                Dtype max_y = 0;
                                if (hh > sy) {
                                    max_y = hh - sy;
                                } else {
                                    max_y = sy - hh;
                                }
                                if (1 - max_y < 0) {
                                    max_y = 0;
                                } else {
                                    max_y = 1 - max_y;
                                }
                                Dtype max_x = 0;
                                if (ww > sx) {
                                    max_x = ww - sx;
                                } else {
                                    max_x = sx - ww;
                                }
                                if (1 - max_x < 0) {
                                    max_x = 0;
                                } else {
                                    max_x = 1 - max_x;
                                }
                                out_data[out->offset(n, c, h, w)] += in->data_at(n, c, hh, ww) * max_x*max_y;
                            }
                        }
                    }
                }
            }
        }
    }

    template void affine_transform(const Blob<float>* in, const Blob<float>* theta, Blob<float>* out);

    template void affine_transform(const Blob<double>* in, const Blob<double>* theta, Blob<double>* out);

    template <typename TypeParam>
    class SpatialTransformerLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
    protected:

        SpatialTransformerLayerTest()
        : blob_data_(new Blob<Dtype>(vector<int>{2, 3, 5, 9})),
        blob_theta_(new Blob<Dtype>(vector<int>{2, 6})),
        blob_top_(new Blob<Dtype>()) {
        }

        virtual void SetUp() {
            FillerParameter filler_param;
            filler_param.set_min(-1);
            filler_param.set_max(1);
            UniformFiller<Dtype> filler(filler_param);
            filler.Fill(this->blob_data_);
            blob_bottom_vec_.push_back(blob_data_);
            blob_bottom_vec_.push_back(blob_theta_);
            blob_top_vec_.push_back(blob_top_);
        }

        virtual ~SpatialTransformerLayerTest() {
            delete blob_data_;
            delete blob_theta_;
            delete blob_top_;
        }

        virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
            this->ref_blob_top_.reset(new Blob<Dtype>());
            this->ref_blob_top_->ReshapeLike(*top);
            return this->ref_blob_top_.get();
        }

        Blob<Dtype> * const blob_data_;
        Blob<Dtype> * const blob_theta_;
        Blob<Dtype> * const blob_top_;
        shared_ptr<Blob<Dtype> > ref_blob_top_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
    };
    
    TYPED_TEST_CASE(SpatialTransformerLayerTest, TestDtypesAndDevices);
    // check top blob shape
    TYPED_TEST(SpatialTransformerLayerTest, TestSetUp) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        EXPECT_EQ(this->blob_top_->num(), this->blob_data_->num());
        EXPECT_EQ(this->blob_top_->channels(), this->blob_data_->channels());
        EXPECT_EQ(this->blob_top_->height(), this->blob_data_->height());
        EXPECT_EQ(this->blob_top_->width(), this->blob_data_->width());
    }

    // test forward: to test flip: 1/(h-1) & 1/(w-1) must be 2^{-n}
    TYPED_TEST(SpatialTransformerLayerTest, TestIdenticalForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        ConstantFiller<Dtype> constant_filler(filler_param);
        constant_filler.Fill(this->blob_theta_);
        this->blob_theta_->mutable_cpu_data()[0] = 1.;
        this->blob_theta_->mutable_cpu_data()[1] = 0.;
        this->blob_theta_->mutable_cpu_data()[2] = 0.;
        this->blob_theta_->mutable_cpu_data()[3] = 0.;
        this->blob_theta_->mutable_cpu_data()[4] = 1.;
        this->blob_theta_->mutable_cpu_data()[5] = 0.;
        this->blob_theta_->mutable_cpu_data()[0 + 6] = 1.;
        this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[2 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[4 + 6] = 1.;
        this->blob_theta_->mutable_cpu_data()[5 + 6] = 0.;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->blob_data_->cpu_data();
        for (int i = 0; i < this->blob_top_->count(); ++i) {
            EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
        }
    }

    TYPED_TEST(SpatialTransformerLayerTest, TestFlipXForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        ConstantFiller<Dtype> constant_filler(filler_param);
        constant_filler.Fill(this->blob_theta_);
        this->blob_theta_->mutable_cpu_data()[0] = -1.;
        this->blob_theta_->mutable_cpu_data()[1] = 0.;
        this->blob_theta_->mutable_cpu_data()[2] = 0.;
        this->blob_theta_->mutable_cpu_data()[3] = 0.;
        this->blob_theta_->mutable_cpu_data()[4] = 1.;
        this->blob_theta_->mutable_cpu_data()[5] = 0.;
        this->blob_theta_->mutable_cpu_data()[0 + 6] = -1.;
        this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[2 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[4 + 6] = 1.;
        this->blob_theta_->mutable_cpu_data()[5 + 6] = 0.;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->blob_data_->cpu_data();
        int num = this->blob_top_->num();
        int channels = this->blob_top_->channels();
        int height = this->blob_top_->height();
        int width = this->blob_top_->width();
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)], ref_top_data[this->blob_data_->offset(n, c, h, width - 1 - w)], 1e-4);
                    }
                }
            }
        }
    }

    TYPED_TEST(SpatialTransformerLayerTest, TestFlipYForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        ConstantFiller<Dtype> constant_filler(filler_param);
        constant_filler.Fill(this->blob_theta_);
        this->blob_theta_->mutable_cpu_data()[0] = (Dtype) 1.;
        this->blob_theta_->mutable_cpu_data()[1] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[2] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[3] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[4] = (Dtype) - 1.;
        this->blob_theta_->mutable_cpu_data()[5] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[0 + 6] = (Dtype) 1.;
        this->blob_theta_->mutable_cpu_data()[1 + 6] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[2 + 6] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[3 + 6] = (Dtype) 0.;
        this->blob_theta_->mutable_cpu_data()[4 + 6] = (Dtype) - 1.;
        this->blob_theta_->mutable_cpu_data()[5 + 6] = (Dtype) 0.;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->blob_data_->cpu_data();
        int num = this->blob_top_->num();
        int channels = this->blob_top_->channels();
        int height = this->blob_top_->height();
        int width = this->blob_top_->width();
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)], ref_top_data[this->blob_data_->offset(n, c, height - 1 - h, w)], 1e-4);
                    }
                }
            }
        }
    }

    TYPED_TEST(SpatialTransformerLayerTest, TestFlipXYForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        ConstantFiller<Dtype> constant_filler(filler_param);
        constant_filler.Fill(this->blob_theta_);
        this->blob_theta_->mutable_cpu_data()[0] = -1.;
        this->blob_theta_->mutable_cpu_data()[1] = 0.;
        this->blob_theta_->mutable_cpu_data()[2] = 0.;
        this->blob_theta_->mutable_cpu_data()[3] = 0.;
        this->blob_theta_->mutable_cpu_data()[4] = -1.;
        this->blob_theta_->mutable_cpu_data()[5] = 0.;
        this->blob_theta_->mutable_cpu_data()[0 + 6] = -1.;
        this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[2 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[4 + 6] = -1.;
        this->blob_theta_->mutable_cpu_data()[5 + 6] = 0.;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->blob_data_->cpu_data();
        int num = this->blob_top_->num();
        int channels = this->blob_top_->channels();
        int height = this->blob_top_->height();
        int width = this->blob_top_->width();
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)], ref_top_data[this->blob_data_->offset(n, c, height - 1 - h, width - 1 - w)], 1e-4);
                    }
                }
            }
        }
    }

    TYPED_TEST(SpatialTransformerLayerTest, TestScalingForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        ConstantFiller<Dtype> constant_filler(filler_param);
        constant_filler.Fill(this->blob_theta_);
        this->blob_theta_->mutable_cpu_data()[0] = 2;
        this->blob_theta_->mutable_cpu_data()[1] = 0.;
        this->blob_theta_->mutable_cpu_data()[2] = 1.;
        this->blob_theta_->mutable_cpu_data()[3] = 0.;
        this->blob_theta_->mutable_cpu_data()[4] = 2;
        this->blob_theta_->mutable_cpu_data()[5] = 1.;

        this->blob_theta_->mutable_cpu_data()[0 + 6] = 2;
        this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[2 + 6] = 1.;
        this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
        this->blob_theta_->mutable_cpu_data()[4 + 6] = 2;
        this->blob_theta_->mutable_cpu_data()[5 + 6] = 1.;
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->blob_data_->cpu_data();
        int num = this->blob_top_->num();
        int channels = this->blob_top_->channels();
        int height = this->blob_top_->height();
        int width = this->blob_top_->width();
        for (int n = 0; n < num / 2; ++n) {
            for (int c = 0; c < channels / 2; ++c) {
                for (int h = 0; h < height / 2; ++h) {
                    for (int w = 0; w < width / 2; ++w) {
                        EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)], ref_top_data[this->blob_data_->offset(n, c, h * 2, w * 2)], 1e-4);
                    }
                }
            }
        }
    }

    TYPED_TEST(SpatialTransformerLayerTest, TestAffineForward) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        filler_param.set_min(-1);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_theta_);
        LayerParameter layer_param;
        shared_ptr<Layer<Dtype> > layer(
                new SpatialTransformerLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        affine_transform(this->blob_data_, this->blob_theta_, this->MakeReferenceTop(this->blob_top_));
        const Dtype* top_data;
        const Dtype* ref_top_data;
        top_data = this->blob_top_->cpu_data();
        ref_top_data = this->ref_blob_top_->cpu_data();
        for (int i = 0; i < this->blob_top_->count(); ++i) {
            EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
        }
    }

    // test gradients of data part using standard caffe utility 
    TYPED_TEST(SpatialTransformerLayerTest, TestDataGradient) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        filler_param.set_min(-1);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_theta_);
        LayerParameter layer_param;
        SpatialTransformerLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-3);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                this->blob_top_vec_, 0);
    }
    // finite difference with mask trick for max operation: track the winner (refer to http://cs231n.github.io/neural-networks-3/)
    template <typename Dtype>
    void theta_gradient(const Blob<Dtype>* in, const Blob<Dtype>* theta, double delta, Blob<Dtype>* gradient) {
        int num = in->shape(0);
        int channels = in->shape(1);
        int height = in->shape(2);
        int width = in->shape(3);
        Dtype* gradient_data = gradient->mutable_cpu_diff();
        caffe_set<Dtype>(theta->count(), 0, gradient_data);
        const Dtype* theta_data = theta->cpu_data();
        for (int i = 0; i < 6; ++i) {
            for (int n = 0; n < num; ++n) {
                for (int h = 0; h < height; ++h) {
                    double ty = h / (double) (height - 1) * (double) 2. - (double) 1.;
                    for (int w = 0; w < width; ++w) {
                        double tx = w / (double) (width - 1)*(double) 2. - (double) 1.;
                        double sx = tx * theta_data[n * 6] + ty * theta_data[n * 6 + 1] + theta_data[n * 6 + 2];
                        double sy = tx * theta_data[n * 6 + 3] + ty * theta_data[n * 6 + 4] + theta_data[n * 6 + 5];
                        double sxn = sx;
                        double syn = sy;
                        if (i == 0) {
                            sxn += delta * tx;
                        } else if (i == 1) {
                            sxn += delta * ty;
                        } else if (i == 2) {
                            sxn += delta;
                        } else if (i == 3) {
                            syn += delta * tx;
                        } else if (i == 4) {
                            syn += delta * ty;
                        } else if (i == 5) {
                            syn += delta;
                        }
                        sx = (sx + 1.) / (double) 2. * (width - 1);
                        sy = (sy + 1.) / (double) 2. * (height - 1);
                        sxn = (sxn + 1.) / (double) 2. * (width - 1);
                        syn = (syn + 1.) / (double) 2. * (height - 1);
                        for (int c = 0; c < channels; ++c) {
                            for (int hh = 0; hh < height; ++hh) {
                                for (int ww = 0; ww < width; ++ww) {
                                    double max_y = 0;
                                    double max_yn = 0;
                                    if (hh > sy) {
                                        max_y = hh - sy;
                                        max_yn = hh - syn;
                                    } else {
                                        max_y = sy - hh;
                                        max_yn = syn - hh;
                                    }
                                    if (1 - max_y < 0) {
                                        max_y = 0;
                                        max_yn = 0;
                                    } else {
                                        max_y = 1 - max_y;
                                        max_yn = 1 - max_yn;
                                    }
                                    double max_x = 0;
                                    double max_xn = 0;
                                    if (ww > sx) {
                                        max_x = ww - sx;
                                        max_xn = ww - sxn;
                                    } else {
                                        max_x = sx - ww;
                                        max_xn = sxn - ww;
                                    }
                                    if (1 - max_x < 0) {
                                        max_x = 0;
                                        max_xn = 0;
                                    } else {
                                        max_x = 1 - max_x;
                                        max_xn = 1 - max_xn;
                                    }
                                    gradient_data[i + n * 6] += (Dtype) (in->data_at(n, c, hh, ww) * (max_xn * max_yn - max_x * max_y));
                                }
                            }
                        }
                    }
                }
            }
        }
        // to improve numeric precision
        for(int i = 0; i < theta->count(); ++i) {
            gradient_data[i] /= delta;
        }
    }

    template void theta_gradient(const Blob<float>* data, const Blob<float>* theta, double delta, Blob<float>* gradient);

    template void theta_gradient(const Blob<double>* data, const Blob<double>* theta, double delta, Blob<double>* gradient);


    // test gradients of theta using finite difference method: max operator would fail caffe utility
    TYPED_TEST(SpatialTransformerLayerTest, TestThetaGradient) {
        typedef typename TypeParam::Dtype Dtype;
        FillerParameter filler_param;
        filler_param.set_min(-1);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_theta_);
        LayerParameter layer_param;
        SpatialTransformerLayer<Dtype> layer(layer_param);
        // call backward to generate theta_gradient
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        for (int i = 0; i < this->blob_top_->count(); ++i) {
            this->blob_top_->mutable_cpu_diff()[i] = (Dtype) 1.;
        }
        vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
        layer.Backward(this->blob_top_vec_, propagate_down,
                this->blob_bottom_vec_);
        // compute theta gradient using finite difference
        shared_ptr<Blob<Dtype> > ref_blob_theta_diff;
        ref_blob_theta_diff.reset(new Blob<Dtype>());
        ref_blob_theta_diff->ReshapeLike(*(this->blob_theta_));
        theta_gradient(this->blob_data_, this->blob_theta_, (double) 1e-4, ref_blob_theta_diff.get());
        const Dtype* theta_diff = this->blob_theta_->cpu_diff();
        const Dtype* ref_theta_diff = ref_blob_theta_diff->cpu_diff();
        for (int i = 0; i < this->blob_theta_->count(); ++i) {
            EXPECT_NEAR(theta_diff[i], ref_theta_diff[i], 1e-4) << "i=" << i;
        }
    }
}
