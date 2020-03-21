#include <cstdio>
#include <iostream>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

// ---------------- 注册Ops ------------------
REGISTER_OP("TrilinearDevoxelize")
    .Attr("resolution: int")
    .Attr("training: bool")
	.Input("coords: float32")
	.Input("features: float32")
	.Output("outs: float32")
	.Output("inds: int32")
	.Output("wgts: float32");
REGISTER_OP("TrilinearDevoxelizeGrad")
    .Attr("resolution: int")
	.Input("grad_outs: float32")
	.Input("indices: int32")
	.Input("weights: float32")
	.Output("grad_features: float32");

using namespace std;

//void TrilinearDevoxelizeKernelLauncher(int b, int c, int n, int r1, int r2, int r3, bool training, const float *xyz, const float *xyz, int *idx, float *weight, float *outxyz);

void trilinearDevoxelizeKernelLauncher(int b, int c, int n, int r, int r2, int r3,
                          bool training, const float *coords, const float *feat,
                          int *inds, float *wgts, float *outs);


// ----------------- Kernel的创建与调用 ---------------
// 写自定义op类接口，继承tensorflow::OpKernel基类并重写其Compute方法；
class TrilinearDevoxelizeGpuOp : public OpKernel{
    public:
        // explicit关键字用来修饰类的构造函数
        // CreateOpKernel
        explicit TrilinearDevoxelizeGpuOp(OpKernelConstruction* context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution_));
            OP_REQUIRES(context, resolution_ > 0, errors::InvalidArgument("Devoxelize expects positive resolution"));
            // OP_REQUIRES_OK 分配新的内存空间
            OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
            OP_REQUIRES(context, training_ = (true || false), errors::InvalidArgument("Devoxelize expects positive resolution"));}
        // 创建OpKernelContext, OpKernel::Compute调用
        void Compute(OpKernelContext * context)override{
            // 获取输入 tensor(指针)
            const Tensor& coords_tensor=context->input(0);
            auto coords_flat=coords_tensor.flat<float>();
            const float *coords = &(coords_flat(0));
            OP_REQUIRES(context, coords_tensor.dims()==3, errors::InvalidArgument("TrilinearDevoxelize requires coords be of shape (batch,#points,3)"));
            OP_REQUIRES(context,coords_tensor.shape().dim_size(1)==3,errors::InvalidArgument("TrilinearDevoxelize only accepts 3d point set coords"));

            const Tensor& features_tensor = context->input(1);
            auto features_flat = features_tensor.flat<float>();
            const float *features = &(features_flat(0));

            int b = features_tensor.shape().dim_size(0);
            int c = features_tensor.shape().dim_size(1);
            int n = coords_tensor.shape().dim_size(2);
            int r1 = resolution_;
            int r2 = r1 * r1;
            int r3 = r2 * r1;
            // 创建一个输出 tensor
            Tensor * outs_tensor = nullptr;
            Tensor * inds_tensor = nullptr;
            Tensor * wgts_tensor = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,c,n},&outs_tensor));
            auto out_flat = outs_tensor->flat<float>();
            float *outpc = &out_flat(0);

            if(training_ = true){
                OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,8,n},&inds_tensor));
                OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,8,n},&wgts_tensor));
            }else{
                OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{1},&inds_tensor));
                OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{1},&wgts_tensor));
            }
            auto inds_flat = inds_tensor->flat<int>();
            auto wgts_flat = wgts_tensor->flat<float>();
            int *inds = &inds_flat(0);
            float *wgts = &wgts_flat(0);
            // Call the cuda kernel launcher
            trilinearDevoxelizeKernelLauncher(b,c,n,r1,r2,r3,training_,coords,features,inds,wgts,outpc);
        }
    private:
        int resolution_;
        bool training_;
    };

// ---------------- 注册Kernel ------------------
REGISTER_KERNEL_BUILDER(Name("TrilinearDevoxelize").Device(DEVICE_GPU), TrilinearDevoxelizeGpuOp);



void trilinearDevoxelizeKernelGradLauncher(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_outs,
                               float *grad_features);

class TrilinearDevoxelizeGradOp:public OpKernel{
    public:
        explicit TrilinearDevoxelizeGradOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution_));
            OP_REQUIRES(context, resolution_ > 0, errors::InvalidArgument("Devoxelize expects positive resolution"));}
        // OpKernelContext获取输出，分配输出
        void Compute(OpKernelContext * context) override{
            const Tensor& grad_outs_tensor = context->input(0);
            OP_REQUIRES(context, grad_outs_tensor.dims()==3, errors::InvalidArgument("TrilinearDevoxelize requires grad_outs be of shape (batch,c,#points)"));
            auto grad_outs_flat = grad_outs_tensor.flat<float>();
            const float *grad_outs = &(grad_outs_flat(0));

            int b = grad_outs_tensor.shape().dim_size(0);
            int c = grad_outs_tensor.shape().dim_size(1);
            int n = grad_outs_tensor.shape().dim_size(2);
            int r = resolution_;
            int r3 = r * r * r;

            const Tensor& indices_tensor = context->input(1);
            OP_REQUIRES(context, indices_tensor.dims()==3, errors::InvalidArgument("TrilinearDevoxelize requires indices be of shape (batch,8,#points)"));
            auto indices_flat = indices_tensor.flat<int>();
            const int * indices = &(indices_flat(0));

            const Tensor& weights_tensor = context->input(2);
            OP_REQUIRES(context, weights_tensor.dims()==3, errors::InvalidArgument("TrilinearDevoxelize requires weights be of shape (batch,8,#points)"));
            auto weights_flat = weights_tensor.flat<float>();
            const float * weights = &(weights_flat(0));
            // 创建一个输出 tensor
            Tensor * grad_features_tensor = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,c,r3}, &grad_features_tensor));
            auto grad_features_flat = grad_features_tensor->flat<float>();
            float * grad_features = &(grad_features_flat(0));
            trilinearDevoxelizeKernelGradLauncher(b,c,n,r3,indices,weights,grad_outs,grad_features);
        }
    private:
        int resolution_;
};
REGISTER_KERNEL_BUILDER(Name("TrilinearDevoxelizeGrad").Device(DEVICE_GPU),TrilinearDevoxelizeGradOp);

