#!/usr/bin/env bash

nvcc -std=c++11 -c -o deform_conv3d_grad.cu.o deform_conv3d_grad.cu.cc \
    -I/home/sl/anaconda2/envs/tensorflow/lib/python3.4/site-packages/tensorflow/include \
     -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L/usr/local/cuda-8.0/lib64/ --expt-relaxed-constexpr \
     -Wno-deprecated-gpu-targets
g++ -std=c++11 -shared -o deform_conv3d_grad.so deform_conv3d_grad.cc\
    deform_conv3d_grad.cu.o -I/home/sl/anaconda2/envs/tensorflow/lib/python3.4/site-packages/tensorflow/include \
     -fPIC -lcudart -L/usr/local/cuda-8.0/lib64\
     -DGOOGLE_CUDA=1 -Wfatal-errors -I/usr/local/cuda-8.0/include -D_GLIBCXX_USE_CXX11_ABI=0 # 老版本abi





