#!/usr/bin/env bash
#export PATH="/home/sl/anaconda2/bin:$PATH" # -I添加新的目录
#source activate tensorflow
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc -std=c++11 -c -o deform_conv.cu.o deform_conv.cu.cc \
     -I/home/sl/anaconda2/envs/tensorflow/lib/python3.4/site-packages/tensorflow/include \
     -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L/usr/local/cuda-8.0/lib64/ --expt-relaxed-constexpr
g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc\
    deform_conv.cu.o -I/home/sl/anaconda2/envs/tensorflow/lib/python3.4/site-packages/tensorflow/include \
     -fPIC -lcudart -L/usr/local/cuda-8.0/lib64\
     -DGOOGLE_CUDA=1 -Wfatal-errors -I/usr/local/cuda-8.0/include -D_GLIBCXX_USE_CXX11_ABI=0 # 老版本abi
# -wfatal-error 出现第一次错误停止编译
# -I 添加搜索目录
# -D 定义宏
# -L 添加链接库目录，链接库由-l指定
# -fPIC 编译未位置独立的代码
# -expt Allow host code to invoke __device__constexpr functions, and device code to invoke __host__constexpr functions
