
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "dnaseq_beam.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DnaseqLFBS")
    .Attr("beamtail_length: int = 0")
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
        return tensorflow::Status();
    });

REGISTER_OP("DnaseqBeam")
.Attr("number_of_beams: int = 64")
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
    return tensorflow::Status();
    });

REGISTER_OP("DnaseqViterbi")
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
    return tensorflow::Status();
    });


class DnaseqLFBS : public OpKernel {
 public:
  explicit DnaseqLFBS(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("beamtail_length", &beamtail_length_));
  }

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
      int beamtail_length = beamtail_length_; // Should be picked from attribute and not hard coded
      if (beamtail_length < kmer_length) beamtail_length = kmer_length; // Make sure that beamtail_length is at lest kmer_length
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
      DCHECK_EQ(num_kmers, transition_probability_shape.dim_size(1));

      int block_length = nn_prior_shape.dim_size(1);
      int duration_length = duration_probability_shape.dim_size(1);

      // Let user know we are here
      //printf("LFBS search got called\n");
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
      
      dnaseq_lfbs_gpu(stream,
                         (float *) nn_prior_t.data(), (int32_t *) output_t->data(),
                         (float *) duration_probability_t.data(), (float *) tail_factor_t.data(), (float*) transition_probability_t.data(),
                         (int32_t *) beamlist_current_id_t.data(), (int32_t *) beamlist_parent_id_t.data(),
                         (int32_t *) beamlist_x_parent_id_t.data(),
                         (float *) beamlist_duration_t.data(), (float *) beamlist_x_duration_t.data(),
                         (float *) beamlist_score_t.data(), (float *) beamlist_x_score_t.data(),
                         (int32_t *) traceback_t.data(),
                         block_length, kmer_length, beamtail_length, batch_size, duration_length);
   }
   private:
       int beamtail_length_;
};

class DnaseqBeam : public OpKernel {
public:
    explicit DnaseqBeam(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("number_of_beams", &number_of_beams_));
    }

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
        int num_kmers = 1 << (2 * kmer_length);
        int beamlist_length = number_of_beams_;
        int sortlist_length = 1;
        for (int k = 5 * beamlist_length - 1; k > 0; k >>= 1) { // Find minimum length for sorting expanded beam list
            sortlist_length <<= 1;
        }
        int x_factor = sortlist_length > num_kmers ? sortlist_length : num_kmers;

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
        DCHECK_EQ(num_kmers, transition_probability_shape.dim_size(1));

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
        beamlist_current_id_shape.AddDim(beamlist_length);
        Tensor beamlist_current_id_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_current_id_shape, &beamlist_current_id_t));

        TensorShape beamlist_x_current_id_shape;
        beamlist_x_current_id_shape.AddDim(batch_size);
        beamlist_x_current_id_shape.AddDim(x_factor);
        Tensor beamlist_x_current_id_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_x_current_id_shape, &beamlist_x_current_id_t));

        TensorShape beamlist_parent_id_shape;
        beamlist_parent_id_shape.AddDim(batch_size);
        beamlist_parent_id_shape.AddDim(beamlist_length);
        Tensor beamlist_parent_id_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_parent_id_shape, &beamlist_parent_id_t));

        TensorShape beamlist_x_parent_id_shape;
        beamlist_x_parent_id_shape.AddDim(batch_size);
        beamlist_x_parent_id_shape.AddDim(x_factor);
        Tensor beamlist_x_parent_id_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_x_parent_id_shape, &beamlist_x_parent_id_t));

        TensorShape beamlist_duration_shape;
        beamlist_duration_shape.AddDim(batch_size);
        beamlist_duration_shape.AddDim(duration_length);
        beamlist_duration_shape.AddDim(beamlist_length);
        Tensor beamlist_duration_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_duration_shape, &beamlist_duration_t));

        TensorShape beamlist_x_duration_shape;
        beamlist_x_duration_shape.AddDim(batch_size);
        beamlist_x_duration_shape.AddDim(duration_length);
        beamlist_x_duration_shape.AddDim(x_factor);
        Tensor beamlist_x_duration_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_x_duration_shape, &beamlist_x_duration_t));

        TensorShape beamlist_score_shape;
        beamlist_score_shape.AddDim(batch_size);
        beamlist_score_shape.AddDim(beamlist_length);
        Tensor beamlist_score_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_score_shape, &beamlist_score_t));

        TensorShape beamlist_x_score_shape;
        beamlist_x_score_shape.AddDim(batch_size);
        beamlist_x_score_shape.AddDim(x_factor);
        Tensor beamlist_x_score_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, beamlist_x_score_shape, &beamlist_x_score_t));

        TensorShape beamlist_kmer_shape;
        beamlist_kmer_shape.AddDim(batch_size);
        beamlist_kmer_shape.AddDim(block_length);
        beamlist_kmer_shape.AddDim(beamlist_length);
        Tensor beamlist_kmer_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_kmer_shape, &beamlist_kmer_t));

        TensorShape beamlist_x_kmer_shape;
        beamlist_x_kmer_shape.AddDim(batch_size);
        beamlist_x_kmer_shape.AddDim(x_factor);
        Tensor beamlist_x_kmer_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, beamlist_x_kmer_shape, &beamlist_x_kmer_t));

        TensorShape merge_id_shape;
        merge_id_shape.AddDim(batch_size);
        merge_id_shape.AddDim(x_factor);
        Tensor merge_id_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, merge_id_shape, &merge_id_t));

        TensorShape list_pos_shape;
        list_pos_shape.AddDim(batch_size);
        list_pos_shape.AddDim(x_factor);
        Tensor list_pos_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, list_pos_shape, &list_pos_t));

        TensorShape traceback_shape;
        traceback_shape.AddDim(batch_size);
        traceback_shape.AddDim(block_length);
        traceback_shape.AddDim(beamlist_length);
        Tensor traceback_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, traceback_shape, &traceback_t));

        TensorShape traceback_x_shape;
        traceback_x_shape.AddDim(batch_size);
        traceback_x_shape.AddDim(x_factor);
        Tensor traceback_x_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, traceback_x_shape, &traceback_x_t));

        //The next complies, so it will be the base for moving forward
        auto dev = context->eigen_device<Eigen::GpuDevice>();
        auto stream = dev.stream();

        dnaseq_beam_gpu(stream,
            (float*)nn_prior_t.data(), (int32_t*)output_t->data(),
            (float*)duration_probability_t.data(), (float*)tail_factor_t.data(), (float*)transition_probability_t.data(),
            (int32_t*)beamlist_current_id_t.data(), (int32_t*)beamlist_x_current_id_t.data(),
            (int32_t*)beamlist_parent_id_t.data(), (int32_t*)beamlist_x_parent_id_t.data(),
            (int32_t*)beamlist_kmer_t.data(), (int32_t*)beamlist_x_kmer_t.data(),
            (float*)beamlist_duration_t.data(), (float*)beamlist_x_duration_t.data(),
            (float*)beamlist_score_t.data(), (float*)beamlist_x_score_t.data(),
            (int32_t*)merge_id_t.data(), (int32_t*)list_pos_t.data(),
            (int32_t*)traceback_t.data(), (int32_t*)traceback_x_t.data(),
            block_length, kmer_length, beamlist_length, batch_size, duration_length);
    }
    private:
        int number_of_beams_;
};

class DnaseqViterbi : public OpKernel {
public:
    explicit DnaseqViterbi(OpKernelConstruction* context) : OpKernel(context) {}

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
        int num_kmers = 1 << (2 * kmer_length);

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
        DCHECK_EQ(num_kmers, transition_probability_shape.dim_size(1));

        int block_length = nn_prior_shape.dim_size(1);
        int duration_length = duration_probability_shape.dim_size(1);

        // Let user know we are here
        //printf("Viterbi got called\n");
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
    
        TensorShape alpha_shape;
        alpha_shape.AddDim(batch_size);
        alpha_shape.AddDim(block_length);
        alpha_shape.AddDim(num_kmers * duration_length);
        Tensor alpha_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, alpha_shape, &alpha_t));

        TensorShape traceback_shape;
        traceback_shape.AddDim(batch_size);
        traceback_shape.AddDim(block_length);
        traceback_shape.AddDim(num_kmers * duration_length);
        Tensor traceback_t;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, traceback_shape, &traceback_t));

        //The next complies, so it will be the base for moving forward
        auto dev = context->eigen_device<Eigen::GpuDevice>();
        auto stream = dev.stream();

        dnaseq_viterbi_gpu(stream,
            (float*)nn_prior_t.data(), (int32_t*)output_t->data(),
            (float*)duration_probability_t.data(), (float*)tail_factor_t.data(), (float*)transition_probability_t.data(),
            (float*)alpha_t.data(), (int32_t*)traceback_t.data(),
            block_length, kmer_length, batch_size, duration_length);
    }
};

REGISTER_KERNEL_BUILDER(Name("DnaseqLFBS").Device(DEVICE_GPU), DnaseqLFBS);
REGISTER_KERNEL_BUILDER(Name("DnaseqBeam").Device(DEVICE_GPU), DnaseqBeam);
REGISTER_KERNEL_BUILDER(Name("DnaseqViterbi").Device(DEVICE_GPU), DnaseqViterbi);
