#include <iostream>
#include "deform_conv.h"
template <>
struct add<float>{
    float operator()(float a, float b){
        return a+b;
    };
};
template<>
struct add<int>{
    int operator()(int a, int b){
        return a-b;
    };
};


template <typename T>
T test(T a){
    return add<T>()(1,1);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    cout<<test(c);
    cout<<test(1);
    return 0;
}