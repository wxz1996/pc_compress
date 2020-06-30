  #include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("NnDistance")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Output("dist1: float32")
	.Output("idx1: int32")
	.Output("dist2: float32")
	.Output("idx2: int32");
REGISTER_OP("NnDistanceGrad")
	.Input("xyz1: float32")
	.Input("xyz2: float32")
	.Input("grad_dist1: float32")
	.Input("idx1: int32")
	.Input("grad_dist2: float32")
	.Input("idx2: int32")
	.Output("grad_xyz1: float32")
	.Output("grad_xyz2: float32");

using namespace tensorflow;

void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i);
class NnDistanceGpuOp : public OpKernel{
	public:
		explicit NnDistanceGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			const Tensor& xyz2_tensor=context->input(1);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("NnDistance only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("NnDistance only accepts 3d point set xyz2"));
			int m=xyz2_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistance expects xyz1 and xyz2 have same batch size"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			Tensor * dist1_tensor=NULL;
			Tensor * idx1_tensor=NULL;
			Tensor * dist2_tensor=NULL;
			Tensor * idx2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist1_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx1_tensor));
			auto dist1_flat=dist1_tensor->flat<float>();
			auto idx1_flat=idx1_tensor->flat<int>();
			OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,m},&dist2_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,m},&idx2_tensor));
			auto dist2_flat=dist2_tensor->flat<float>();
			auto idx2_flat=idx2_tensor->flat<int>();
			float * dist1=&(dist1_flat(0));
			int * idx1=&(idx1_flat(0));
			float * dist2=&(dist2_flat(0));
			int * idx2=&(idx2_flat(0));
			// Call the cuda kernel launcher
			NmDistanceKernelLauncher(b,n,xyz1,m,xyz2,dist1,idx1,dist2,idx2);
		}
};
REGISTER_KERNEL_BUILDER(Name("NnDistance").Device(DEVICE_GPU), NnDistanceGpuOp);

void NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2);
class NnDistanceGradGpuOp : public OpKernel{
	public:
		explicit NnDistanceGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz1_tensor=context->input(0);
			const Tensor& xyz2_tensor=context->input(1);
			const Tensor& grad_dist1_tensor=context->input(2);
			const Tensor& idx1_tensor=context->input(3);
			const Tensor& grad_dist2_tensor=context->input(4);
			const Tensor& idx2_tensor=context->input(5);
			OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz1 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("NnDistanceGrad only accepts 3d point set xyz1"));
			int b=xyz1_tensor.shape().dim_size(0);
			int n=xyz1_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz2 be of shape (batch,#points,3)"));
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("NnDistanceGrad only accepts 3d point set xyz2"));
			int m=xyz2_tensor.shape().dim_size(1);
			OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistanceGrad expects xyz1 and xyz2 have same batch size"));
			OP_REQUIRES(context,grad_dist1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires grad_dist1 be of shape(batch,#points)"));
			OP_REQUIRES(context,idx1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires idx1 be of shape(batch,#points)"));
			OP_REQUIRES(context,grad_dist2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires grad_dist2 be of shape(batch,#points)"));
			OP_REQUIRES(context,idx2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires idx2 be of shape(batch,#points)"));
			auto xyz1_flat=xyz1_tensor.flat<float>();
			const float * xyz1=&xyz1_flat(0);
			auto xyz2_flat=xyz2_tensor.flat<float>();
			const float * xyz2=&xyz2_flat(0);
			auto idx1_flat=idx1_tensor.flat<int>();
			const int * idx1=&idx1_flat(0);
			auto idx2_flat=idx2_tensor.flat<int>();
			const int * idx2=&idx2_flat(0);
			auto grad_dist1_flat=grad_dist1_tensor.flat<float>();
			const float * grad_dist1=&grad_dist1_flat(0);
			auto grad_dist2_flat=grad_dist2_tensor.flat<float>();
			const float * grad_dist2=&grad_dist2_flat(0);
			Tensor * grad_xyz1_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad_xyz1_tensor));
			Tensor * grad_xyz2_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,3},&grad_xyz2_tensor));
			auto grad_xyz1_flat=grad_xyz1_tensor->flat<float>();
			float * grad_xyz1=&grad_xyz1_flat(0);
			auto grad_xyz2_flat=grad_xyz2_tensor->flat<float>();
			float * grad_xyz2=&grad_xyz2_flat(0);
			NmDistanceGradKernelLauncher(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_dist2,idx2,grad_xyz1,grad_xyz2);
		}
};
REGISTER_KERNEL_BUILDER(Name("NnDistanceGrad").Device(DEVICE_GPU), NnDistanceGradGpuOp);
