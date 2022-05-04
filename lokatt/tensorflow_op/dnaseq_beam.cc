
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "dnaseq_beam.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DnaseqBeam")
    .Input("nn_prior: float32")
    .Input("duration_probability: float32")
    .Input("tail_factor: float32")
    .Input("transition_probability: float32")
    .Output("seqence: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle nn_prior_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &nn_prior_shape));
        shape_inference::DimensionHandle batch_size = c->Dim(nn_prior_shape, 0);
        //c->set_output(0, c->Vector(batch_size));
        return Status::OK();
    });
 
class DnaseqBeam : public OpKernel {
 public:
  explicit DnaseqBeam(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      // Check the number of input vectors
      DCHECK_EQ(4, context->num_inputs());
      
      // Obtain input tensors
      const Tensor& nn_prior_t = context->input(0);
      const Tensor& duration_probability_t = context->input(1);
      const Tensor& tail_factor_t = context->input(2);
      const Tensor& transition_probability_t = context->input(3);

      // kmer dependent dimension sizes
      int kmer_length = 5;
      // OP_REQUIRES_OK(context, context->GetAttr("kmer_length", &kmer_length));
      int num_kmers = 1 << (2*kmer_length);
      int beamtail_length = 5;
      int num_beamtails = 1 << (2*beamtail_length);

      // Input tensor shape verifications
      TensorShape nn_prior_shape = nn_prior_t.shape();
      DCHECK_EQ(nn_prior_shape.dims(), 3);
      TensorShape duration_probability_shape = duration_probability_t.shape();
      DCHECK_EQ(duration_probability_shape.dims(), 2);
      TensorShape tail_factor_shape = tail_factor_t.shape();
      DCHECK_EQ(tail_factor_shape.dims(), 1);
      TensorShape transition_probability_shape = transition_probability_t.shape();
      DCHECK_EQ(transition_probability_shape.dims(), 2);

      int batch_size = nn_prior_shape.dim_size(0);
      DCHECK_EQ(batch_size, duration_probability_shape.dim_size(0));
      DCHECK_EQ(batch_size, tail_factor_shape.dim_size(0));
      
      DCHECK_EQ(4, transition_probability_shape.dim_size(0));
      DCHECK_EQ(num_beamtails, transition_probability_shape.dim_size(1));

      int block_length = nn_prior_shape.dim_size(1);
      int duration_length = duration_probability_shape.dim_size(1);

      // Let user know we are here
      //printf("Beam search got called\n");
      //printf(" batch size : %d\n", batch_size);
      //printf(" block length : %d\n", block_length);
      //printf(" duration length : %d\n", duration_length);
       
      // Create output tensor
      TensorShape output_shape;
      output_shape.AddDim(batch_size);
      output_shape.AddDim(block_length);
      Tensor* output_t = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_t));
      
      // Create temporary tensors for calculations
      TensorShape beamlist_current_id_shape;
      beamlist_current_id_shape.AddDim(batch_size);
      beamlist_current_id_shape.AddDim(num_beamtails);
      Tensor beamlist_current_id_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_current_id_shape, &beamlist_current_id_t));
          
      TensorShape beamlist_parent_id_shape;
      beamlist_parent_id_shape.AddDim(batch_size);
      beamlist_parent_id_shape.AddDim(num_beamtails);
      Tensor beamlist_parent_id_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_parent_id_shape, &beamlist_parent_id_t));

      TensorShape beamlist_x_parent_id_shape;
      beamlist_x_parent_id_shape.AddDim(batch_size);
      beamlist_x_parent_id_shape.AddDim(4 * num_beamtails);
      Tensor beamlist_x_parent_id_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_x_parent_id_shape, &beamlist_x_parent_id_t));

      TensorShape beamlist_duration_shape;
      beamlist_duration_shape.AddDim(batch_size);
      beamlist_duration_shape.AddDim(duration_length);
      beamlist_duration_shape.AddDim(num_beamtails);
      Tensor beamlist_duration_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_duration_shape, &beamlist_duration_t));

      TensorShape beamlist_x_duration_shape;
      beamlist_x_duration_shape.AddDim(batch_size);
      beamlist_x_duration_shape.AddDim(duration_length);
      beamlist_x_duration_shape.AddDim(4 * num_beamtails);
      Tensor beamlist_x_duration_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_x_duration_shape, &beamlist_x_duration_t));

      TensorShape beamlist_score_shape;
      beamlist_score_shape.AddDim(batch_size);
      beamlist_score_shape.AddDim(num_beamtails);
      Tensor beamlist_score_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_score_shape, &beamlist_score_t));
      
      TensorShape beamlist_x_score_shape;
      beamlist_x_score_shape.AddDim(batch_size);
      beamlist_x_score_shape.AddDim(4 * num_beamtails);
      Tensor beamlist_x_score_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_x_score_shape, &beamlist_x_score_t));
      
      TensorShape traceback_shape;
      traceback_shape.AddDim(batch_size);
      traceback_shape.AddDim(block_length);
      traceback_shape.AddDim(num_beamtails);
      Tensor traceback_t;
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, traceback_shape, &traceback_t));
      
      //The next complies, so it will be the base for moving forward
      auto dev = context->eigen_device<Eigen::GpuDevice>();
      auto stream = dev.stream();
      
      dnaseq_beamsearch_gpu(stream,
                         (float *) nn_prior_t.data(), (int32_t *) output_t->data(),
                         (float *) duration_probability_t.data(), (float *) tail_factor_t.data(), (float*) transition_probability_t.data(),
                         (int32_t *) beamlist_current_id_t.data(), (int32_t *) beamlist_parent_id_t.data(),
                         (int32_t *) beamlist_x_parent_id_t.data(),
                         (float *) beamlist_duration_t.data(), (float *) beamlist_x_duration_t.data(),
                         (float *) beamlist_score_t.data(), (float *) beamlist_x_score_t.data(),
                         (int32_t *) traceback_t.data(),
                         block_length, kmer_length, beamtail_length, batch_size, duration_length);
   }
};

REGISTER_KERNEL_BUILDER(Name("DnaseqBeam").Device(DEVICE_GPU), DnaseqBeam);
