find_package(CUDA REQUIRED)
message('')


add_definitions(-DGOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(
        "/home/sl/anaconda2/envs/tensorflow/lib/python3.4/site-packages/tensorflow/include"
        "/usr/local/cuda/include"
        "./"
)

#add_executable(deform deform_conv.cc)
cuda_add_executable(deform_conv deform_conv.cc deform_conv.cu.cc)