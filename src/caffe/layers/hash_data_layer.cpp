#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/hash_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

  template <typename Dtype>
  //this function is used to split a string followed by ' '
  vector<std::string>  HashDataLayer<Dtype>::split(const std::string &s, char delim) {
    vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      elems.push_back(item);
    }
    return elems;
  }
 


template <typename Dtype>
HashDataLayer<Dtype>::~HashDataLayer<Dtype>() {
  this->StopInternalThread();
}

  template <typename Dtype>
  void HashDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
					    const vector<Blob<Dtype>*>& top) {
    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width  = this->layer_param_.image_data_param().new_width();
    const bool is_color  = this->layer_param_.image_data_param().is_color();
    string root_folder = this->layer_param_.image_data_param().root_folder();

    CHECK((new_height == 0 && new_width == 0) ||
	  (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
    // Read the file with filenames and labels
    const string& source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    int label_num = 0;
    for(std::string line; std::getline(infile, line);){
      string filename;
      vector<float> labels;
      std::vector<std::string> x = split(line,' ');
      int n = x.size();
      filename = x[0];
      for (int i=1; i<n; i++){
	labels.push_back(atof(x[i].c_str()));
      }
      if(label_num < labels.size())
	label_num = labels.size();
      lines_.push_back(make_pair(filename, labels));
    }

    //read label
    label_inds_.clear();
    inds_.clear();
    clabel_ = 0; 
    for(int k=0; k < label_num; k++){
      vector<int> pos;
      for(int i = 0; i < lines_.size(); i++){
	vector<float> labels = lines_[i].second;
        if(labels[k] > 0)
	  pos.push_back(i);
      }
      label_inds_.push_back(pos);
      inds_.push_back(0);
      shuffle(label_inds_[k].begin(),label_inds_[k].end());
    }

    if (this->layer_param_.image_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages();
    }
    LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
      lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.image_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
	      << top[0]->channels() << "," << top[0]->height() << ","
	      << top[0]->width() << " ; label num = " << label_num;
    // label
    top[1]->Reshape(batch_size,label_num,1,1);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size,label_num,1,1);
    }
  }

  template <typename Dtype>
  void HashDataLayer<Dtype>::ShuffleImages() {
    //caffe::rng_t* prefetch_rng =
    //  static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    // shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  }

  template  <typename Dtype>
  vector<int> HashDataLayer<Dtype>::GetTripletImages() {
    vector<int> vec_ind;
    vec_ind.clear();
    const int m_batch_size = this->layer_param_.image_data_param().batch_size();
    int count = lines_.size();
    vector<float> labels = lines_[0].second;
    int label_size = labels.size();

    int num_each = m_batch_size / label_size; 
    
    for(int i=0; i<label_size; i++){
      int tmp= 0;
      while(true){
    	int find = rand() % count;
    	std::vector<float> labels = lines_[find].second;
    	if(labels[i] > 0){
    	  vec_ind.push_back(find);
    	  tmp++;
    	}
    	if(tmp >= num_each)
    	  break;
      }
    }
    while(vec_ind.size() < m_batch_size){
      int find = rand() % count;
      vec_ind.push_back(find);
    }
    shuffle(vec_ind.begin(),vec_ind.end());
    return vec_ind;     
  }


// This function is called on prefetch thread
  template <typename Dtype>
  void HashDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int batch_size = image_data_param.batch_size();
    const int new_height = image_data_param.new_height();
    const int new_width = image_data_param.new_width();
    const bool is_color = image_data_param.is_color();
    string root_folder = image_data_param.root_folder();

    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // Use data_transformer to infer the expected blob shape from a cv_img.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    // datum scales
    const int lines_size = lines_.size();
    vector<int> vec_ind = GetTripletImages();
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[vec_ind[item_id]].first,
					new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[vec_ind[item_id]].first;
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      //prefetch_label[item_id] = lines_[lines_id_].second;
      vector<float> labels = lines_[vec_ind[item_id]].second;
      int label_num = labels.size();
      for (int k = 0; k < label_num; ++k){
	prefetch_label[item_id*label_num+k] = labels[k];
      }

      // go to the next iter
      lines_id_++;
      if (lines_id_ >= lines_size) {
	// We have reached the end. Restart from the first.
	DLOG(INFO) << "Restarting data prefetching from start.";
	lines_id_ = 0;
	if (this->layer_param_.image_data_param().shuffle()) {
	  // ShuffleImages();
	}
      }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  }

  INSTANTIATE_CLASS(HashDataLayer);
  REGISTER_LAYER_CLASS(HashData);

}  // namespace caffe
#endif  // USE_OPENCV
