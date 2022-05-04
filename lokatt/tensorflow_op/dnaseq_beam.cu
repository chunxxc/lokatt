
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

#include "dnaseq_beam.h"

// CUDA constants
//constexpr int max_threads_per_block = 1024;
constexpr int max_threads_per_block = 512;

// Logt sum exp function (Jacobian log) to generate stable probabilities
__device__
float lse(float a, float b) {
    return fmaxf(a, b) + logf(1.0f + expf(-fabsf(a - b)));
}

constexpr float flt_large = 1.0e38f;

__device__
void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] = lse(sdata[tid], sdata[tid + 16]);
    sdata[tid] = lse(sdata[tid], sdata[tid + 8]);
    sdata[tid] = lse(sdata[tid], sdata[tid + 4]);
    sdata[tid] = lse(sdata[tid], sdata[tid + 2]);
    sdata[tid] = lse(sdata[tid], sdata[tid + 1]);
}

__device__
void blockReduce(volatile float* sdata, int tid) {
    if (tid < 32) { // Only run this part with threads in first warp (limiting maximum number of threads to 1024)
        for (int range = 16; range > 0; range >>= 1) {
            int dst = tid * 32;
            int src = (tid + range) * 32;
            if ((src < blockDim.x) && (dst < blockDim.x)) {
                sdata[dst] = lse(sdata[dst], sdata[src]);
            }
        }
    }
}

__device__
void listReduce(volatile float* sdata, int tid) {
    //__syncthreads();
    warpReduce(sdata, tid);
    __syncthreads();
    blockReduce(sdata, tid);
    __syncthreads();
}


// Kernel...
__global__
void dnaseq_beamsearch_kernel(const float* __restrict__ nn_prior, int32_t* __restrict__ output, float* __restrict__ duration_probability, const float* __restrict__ tail_factor, const float* __restrict__ transition_probability,
    int32_t* __restrict__ current_id, int32_t* __restrict__ parent_id, int32_t* __restrict__ parent_x_id,
    float* __restrict__ beamlist_duration, float* __restrict__ beamlist_x_duration, float* __restrict__ beamlist_score, float* __restrict__ beamlist_x_score,
    int32_t* __restrict__ traceback,
    int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length) {

    static __shared__ float sdata[max_threads_per_block + 16];
    sdata[threadIdx.x + 16] = 0.0f; // Clear list for bondary free reduction

    // Constants
    int batch_item = blockIdx.x;
    int total_kmers = 1 << (2 * kmer_length);
    int total_beamtails = 1 << (2 * beamtail_length);
    float tail = tail_factor[batch_item];

    // Position bufferes appropriately for current batch item
    nn_prior += total_kmers * block_length * batch_item;
    output += block_length * batch_item;
    duration_probability += duration_length * batch_item;

    current_id += total_beamtails * batch_item;
    parent_id += total_beamtails * batch_item;
    parent_x_id += 4 * total_beamtails * batch_item;
    beamlist_duration += total_beamtails * duration_length * batch_item;
    beamlist_x_duration += 4 * total_beamtails * duration_length * batch_item;
    beamlist_score += total_beamtails * batch_item;
    beamlist_x_score += 4 * total_beamtails * batch_item;

    traceback += total_beamtails * block_length * batch_item;

    // Temporary storage
    float log_prob;

    // Compute forward step
    for (int n = 0; n < block_length; n++) {
        if (n == 0) { // Initialize beams
            for (int t = threadIdx.x; t < total_beamtails; t += blockDim.x) {
                parent_id[t] = -1; // No parent in first 
                current_id[t] = t;
                for (int d = 0; d < duration_length; d++) {
                    beamlist_duration[t + d * total_beamtails] = 0.0f; // Log prob
                }
                beamlist_score[t] = logf((float) duration_length);
            }
        }
        else {
            for (int t = threadIdx.x; t < total_beamtails; t += blockDim.x) {
                // Set up default traceback
                traceback[t + n * total_beamtails] = t; // Traceback to parent node
                float dur_sum;

                // Expand all children
                for (int b = 0; b < 4; b++) {
                    int tt = t + b * total_beamtails;
                    dur_sum = -flt_large;
                    for (int d = 0; d < duration_length; d++) {
                        log_prob = logf(transition_probability[tt]) + beamlist_duration[t + 0 * total_beamtails] + logf(duration_probability[d]);
                        beamlist_x_duration[tt + d * 4 * total_beamtails] = log_prob;
                        dur_sum = lse(dur_sum, log_prob);
                    }            
                    parent_x_id[tt] = current_id[t];
                    beamlist_x_score[tt] = dur_sum;
                }
                // Propagate parent
                dur_sum = -flt_large;
                for (int d = 0; d < duration_length - 2; d++) {
                    log_prob = beamlist_duration[t + (d + 1) * total_beamtails];
                    beamlist_duration[t + d * total_beamtails] = log_prob;
                    dur_sum = lse(dur_sum, log_prob);
                }
                log_prob = logf(1 - tail) + beamlist_duration[t + (duration_length - 1) * total_beamtails];
                beamlist_duration[t + (duration_length - 2) * total_beamtails] = log_prob;
                dur_sum = lse(dur_sum, log_prob);
                log_prob = logf(tail) + beamlist_duration[t + (duration_length - 1) * total_beamtails];
                beamlist_duration[t + (duration_length - 1) * total_beamtails] = log_prob;
                dur_sum = lse(dur_sum, log_prob);
                beamlist_score[t] = dur_sum;
            }

            // Synchronize after all expanded beams
            __syncthreads();

            for (int t = threadIdx.x; t < total_beamtails; t += blockDim.x) {
                // Check for possible mergers
                for (int b = 0; b < 4; b++) {
                    int tt = t * 4 + b;
                    if (parent_x_id[tt] == parent_id[t]) { // If two beams share the same parent they are the same and can be merged (keep ids)
                        if (beamlist_x_score[tt] > beamlist_score[t]) {
                            traceback[t + n * total_beamtails] = (tt & (total_beamtails - 1)) - total_beamtails; // Negative index for new base
                        }
                        for (int d = 0; d < duration_length; d++) {
                            beamlist_duration[t + d * total_beamtails] = lse(beamlist_duration[t + d * total_beamtails], beamlist_x_duration[tt + d * 4 * total_beamtails]);
                        }
                        beamlist_score[t] = lse(beamlist_score[t], beamlist_x_score[tt]);
                    }
                }
                for (int b = 0; b < 4; b++) {
                    int tt = t * 4 + b; // Check all candidates for exchange
                    if (beamlist_x_score[tt] > beamlist_score[t]) { // Replace beam with better beam
                        parent_id[t] = parent_x_id[tt]; // Inherit parent at replace
                        current_id[t] = n * total_beamtails + t; // Create a new unique id for beam
                        for (int d = 0; d < duration_length; d++) {
                            beamlist_duration[t + d * total_beamtails] =
                                beamlist_x_duration[tt + d * 4 * total_beamtails];
                        }
                        beamlist_score[t] = beamlist_x_score[tt];
                        traceback[t + n * total_beamtails] = (tt & (total_beamtails - 1)) - total_beamtails; // Negative index for new base
                    }
                }
            }
        }

        // Synchronize after all mergers and pruning
        __syncthreads();

        // Update with observation score
        float thread_sum = -flt_large;
        for (int t = threadIdx.x; t < total_beamtails; t += blockDim.x) {
            int k = t >> (2 * (beamtail_length - kmer_length));
            float obs = logf(nn_prior[k + n * total_kmers]);
            float dur_sum = -flt_large;
            for (int d = 0; d < duration_length; d++) {
                log_prob = beamlist_duration[t + d * total_beamtails] + obs;
                beamlist_duration[t + d * total_beamtails] = log_prob;
                dur_sum = lse(dur_sum, log_prob);
            }
            beamlist_score[t] = dur_sum;
            thread_sum = lse(thread_sum, dur_sum);
        }

        // Normalize using shared memory (for numerical stability)
        sdata[threadIdx.x] = thread_sum;
        listReduce(sdata, threadIdx.x);
        thread_sum = sdata[0];
        for (int t = threadIdx.x; t < total_beamtails; t += blockDim.x) {
            for (int d = 0; d < duration_length; d++) {
                beamlist_duration[t + d * total_beamtails] -= thread_sum;
            }
            beamlist_score[t] -= thread_sum;
        }
    }

    // Backtracking
    __syncthreads();
    if (threadIdx.x == 0) {

        // Find the best end beam
        int best_beam = 0;
        float best_score = -flt_large;
        for (int t = 0; t < total_beamtails; t++) {
            if (beamlist_score[t] > best_score) {
                best_beam = t;
                best_score = beamlist_score[t];
            }
        }

        // Backtrack
        int t = best_beam;
        int len = 0;
        for (int n = block_length - 1; n >= 0; n--) {
            int tn = traceback[t + n * total_beamtails];
            if (tn < 0) {
                output[len++] = t >> (2 * (beamtail_length - kmer_length)); // Return kmers
                t = tn + total_beamtails; // Restore index
            }
            else {
                t = tn;
            }
        }
        // Pad with stop symbols
        for (int n = len; n < block_length; n++) {
            output[n] = -1;
        }
        // Reverse sequence
        for (int n = 0; n < (len >> 1); n++) {
            int32_t tmp = output[n];
            output[n] = output[len-n-1];
            output[len - n - 1] = tmp;
        }
    }
}


void dnaseq_beamsearch_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    int32_t* current_id, int32_t* parent_id, int32_t* parent_x_id, float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
    int32_t* traceback,
    int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length) {

    cudaError_t status;

    int total_kmers = 1 << (2 * kmer_length);
    int total_beamtails = 1 << (2 * beamtail_length);

    int threads_per_block = total_beamtails;
    if (threads_per_block > max_threads_per_block) threads_per_block = max_threads_per_block;

    dnaseq_beamsearch_kernel <<< batch_size, threads_per_block, 0, stream >>> (nn_prior, output, duration_probability, tail_factor, transition_probability,
        current_id, parent_id, parent_x_id, beamlist_duration, beamlist_x_duration, beamlist_score, beamlist_x_score,
        traceback, block_length, kmer_length, beamtail_length, batch_size, duration_length);
    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if (status) {
        std::cout << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    }
}
