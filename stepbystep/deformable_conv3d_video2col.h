//
// Created by sl on 8/2/17.
//

#ifndef DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H
#define DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H
template <typename Device, typename T>
struct DeformableConv3dVideo2colFunctor {
    void operator()(const Device& d, int size, const T* in, T* out);
};
#endif //DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H
