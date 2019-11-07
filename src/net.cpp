/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *   
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems. 
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include "net.h"

Net::Net() {
  nets_.clear();
  
  input_blobs_.clear();
  output_blobs_.clear();

  input_plugs_.clear();
  output_plugs_.clear();

  father_ = nullptr;
}

Net::~Net() {
  nets_.clear();
  
  input_blobs_.clear();
  output_blobs_.clear();

  input_plugs_.clear();
  output_plugs_.clear();

  params_.clear();
}

void Net::SetUp() {
  input_blobs_.clear();
  output_blobs_.clear();

  input_plugs_.clear();
  output_plugs_.clear();
  
  nets_.clear();

  params_.clear();
}

//注意：Execute()在类声明中已经被声明为“纯虚函数”。
void Net::Execute() {
  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "CheckInput begins <<<  " << std::endl;
  // 1. check input blobs；看各个input_blobs_的data_（是一个智能指针）是否已经被初始化。
  CheckInput();
  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "CheckInput ends   <<<  " << std::endl;


  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "son net Execute begins <<<  " << std::endl;
  // 2. execute
  for (int i = 0; i < nets_.size(); ++ i) {
	// chg 测-----------------------------------------------------------------------------------------
	//std::cout << "son loop begins <<<  " << std::endl;
    nets_[i]->Execute();
	// chg 测-----------------------------------------------------------------------------------------
	//std::cout << "son loop ends <<<  " << std::endl;
  }
  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "son net Execute ends <<<  " << std::endl;
  


  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "CheckOutput begins <<<  " << std::endl;
  // 3. check output blobs
  CheckOutput();
  // chg 测-----------------------------------------------------------------------------------------
  //std::cout << "CheckOutput done <<<  " << std::endl;

  
}

void Net::CheckInput() {
  for (int i = 0; i < input_blobs_.size(); ++ i) {
    if (input_blobs_[i].data() == nullptr) {
      LOG(INFO) << "Net input haven't been initialized completely!";
      exit(0);
    }
  }
}

// 将output_blobs_的数据转移到output_plugs_数组中（有点复杂，看下面定义中的注释）
// Release掉input_blobs_和output_blobs_
void Net::CheckOutput() {

  // chg 测
  //std::cout << "function CheckOutput() is working......." << std::endl;

  for (int i = 0; i < input_blobs_.size(); ++ i) {
    input_blobs_[i].Release();
  }
  for (int i = 0; i < output_blobs_.size(); ++ i) {
    // connecting output plugs； 把output_plugs_[i]（这是一个Blob指针数组）里的元素所指的Blob对象全部变为output_blobs_[i]
    for (std::vector<Blob*>::iterator blob = output_plugs_[i].begin();
        blob != output_plugs_[i].end(); ++ blob) {
      (*blob)->SetData(output_blobs_[i]);
    }
    // release output blobs
    if (output_plugs_[i].size() != 0) {
      output_blobs_[i].Release();
    }
  } 
}

