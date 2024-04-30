
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

#include "dnaseq_beam.h"

// CUDA constants
constexpr int max_threads_per_block = 1024;

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
    __syncthreads(); // Why is this needed?
    warpReduce(sdata, tid);
    __syncthreads();
    blockReduce(sdata, tid);
    __syncthreads();
}


// Swap 2 values in memory
template <typename T> __device__ void inline swap(T& a, T& b)
{
    T c(a); a = b; b = c;
}
// Bitonic sorting algorithm from Wikipedia
template <typename T> __device__ void inline sort(T* val, int32_t* pos, int length) {
    __syncthreads(); // Assign positions
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        pos[i] = (int32_t) i;
    }
    __syncthreads();
    for (int k = 2; k <= length; k <<= 1) { // k is doubled every iteration
        for (int j = k >> 1; j > 0; j >>= 1) { // j is halved at every iteration, with truncation of fractional parts
            for (int i = threadIdx.x; i < length; i += blockDim.x) {
                int l = i ^ j; // Bitwise xor
                if (l > i) {
                    if ((((i & k) == 0) & (val[i] < val[l])) | (((i & k) != 0) & (val[i] > val[l]))) {
                        swap(val[i], val[l]);
                        swap(pos[i], pos[l]);
                    }
                }
            }
            __syncthreads();
        }
    }
}


// Local focus beam search CUDA kernel
__global__
void dnaseq_lfbs_kernel(const float* __restrict__ nn_prior, int32_t* __restrict__ output, float* __restrict__ duration_probability, const float* __restrict__ tail_factor, const float* __restrict__ transition_probability,
    int32_t* __restrict__ current_id, int32_t* __restrict__ parent_id, int32_t* __restrict__ parent_x_id,
    float* __restrict__ beamlist_duration, float* __restrict__ beamlist_x_duration, float* __restrict__ beamlist_score, float* __restrict__ beamlist_x_score,
    int32_t* __restrict__ traceback,
    int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length) {

    static __shared__ float sdata[max_threads_per_block + 16];
    sdata[threadIdx.x + 16] = -flt_large; // Clear list for bondary free reduction

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
                    int tk = t >> (2 * (beamtail_length - kmer_length));
                    dur_sum = -flt_large;
                    for (int d = 0; d < duration_length; d++) { // *** there is probably an error in the the way that the transition probabilities are handled here *** (assumes kmer...)
                        log_prob = logf(transition_probability[tk]) + beamlist_duration[t + 0 * total_beamtails] + logf(duration_probability[d]);
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
                        //if (current_id[tt] < current_id[t]) { // Keep the lowest id at merge
                        //    current_id[t] = current_id[tt];
                        //}
                    }
                }
                for (int b = 0; b < 4; b++) {
                    int tt = t * 4 + b; // Check all candidates for exchange
                    if (beamlist_x_score[tt] > beamlist_score[t]) { // Replace beam with better beam
                        parent_id[t] = parent_x_id[tt]; // Inherit parent at replace
                        //current_id[t] = current_id[tt]; // Inherit id at replace
                        current_id[t] = n * total_beamtails + t; // Create a new unique id for beam when needed
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


// Full beam search CUDA kernel
__global__
void dnaseq_beam_kernel(const float* __restrict__ nn_prior, int32_t* __restrict__ output, float* __restrict__ duration_probability, const float* __restrict__ tail_factor, const float* __restrict__ transition_probability,
    int32_t* __restrict__ current_id, int32_t* __restrict__ current_x_id, int32_t* __restrict__ parent_id, int32_t* __restrict__ parent_x_id,
    int32_t* __restrict__ beamlist_kmer, int32_t* __restrict__ beamlist_x_kmer,
    float* __restrict__ beamlist_duration, float* __restrict__ beamlist_x_duration, float* __restrict__ beamlist_score, float* __restrict__ beamlist_x_score,
    int32_t* __restrict__ merge_id, int32_t* __restrict__ list_pos,
    int32_t* __restrict__ traceback, int32_t* __restrict__ traceback_x,
    int block_length, int kmer_length, int beamlist_length, int batch_size, int duration_length) {

    static __shared__ float sdata[max_threads_per_block + 16];
    sdata[threadIdx.x + 16] = -flt_large; // Clear list for bondary free reduction

    // Constants
    int batch_item = blockIdx.x;
    int total_kmers = 1 << (2 * kmer_length);
    float tail = tail_factor[batch_item];
    int sortlist_length = 1;
    for (int k = 5 * beamlist_length - 1; k > 0; k >>= 1) { // Find minimum length for sorting expanded beam list
        sortlist_length <<= 1;
    }
    int x_factor = sortlist_length > total_kmers ? sortlist_length : total_kmers;

    // Position bufferes appropriately for current batch item
    nn_prior += total_kmers * block_length * batch_item;
    output += block_length * batch_item;
    duration_probability += duration_length * batch_item;

    current_id += beamlist_length * batch_item;
    current_x_id += x_factor * batch_item;
    parent_id += beamlist_length * batch_item;
    parent_x_id += x_factor * batch_item;
    beamlist_kmer += beamlist_length * block_length * batch_item;
    beamlist_x_kmer += x_factor * batch_item;
    beamlist_duration += beamlist_length * duration_length * batch_item;
    beamlist_x_duration += x_factor * duration_length * batch_item;
    beamlist_score += beamlist_length * batch_item;
    beamlist_x_score += x_factor * batch_item;

    traceback += beamlist_length * block_length * batch_item;
    traceback_x += x_factor * batch_item;

    merge_id += x_factor * batch_item;
    list_pos += x_factor * batch_item;

    // Temporary storage
    float log_prob;

    // Compute forward step
    for (int n = 0; n < block_length; n++) {

        if (n == 0) { // Initialize beams
            for (int t = threadIdx.x; t < total_kmers; t += blockDim.x) {
                parent_x_id[t] = t; // Makes the parents of first generation unique
                current_x_id[t] = t + x_factor; // Must not overlap with parents
                traceback_x[t] = 0; // Initialize

                beamlist_x_score[t] = logf(nn_prior[t + n * total_kmers]);
                beamlist_x_kmer[t] = t;
                for (int d = 0; d < duration_length; d++) {
                    beamlist_x_duration[t + d * x_factor] = beamlist_x_score[t] + logf(1 / (float)duration_length);
                }
            }
        }
        else { // Regular beam expansion
            // Initialize buffers used for sorting
            for (int t = threadIdx.x; t < sortlist_length; t += blockDim.x) {
                beamlist_x_score[t] = -flt_large; // Make sure unused entries end up last
                merge_id[t] = -1; // Make sure unused entries end up last
            }

            for (int t = threadIdx.x; t < beamlist_length; t += blockDim.x) {
                float dur_sum;

                // Continue parent without expanding
                current_x_id[t] = current_id[t];
                parent_x_id[t] = parent_id[t];
                int kp = beamlist_kmer[t + (n - 1) * beamlist_length]; // Parent (non-exanded) kmer
                beamlist_x_kmer[t] = kp;
                dur_sum = -flt_large;
                float obs = logf(nn_prior[kp + n * total_kmers]);
                for (int d = 0; d < duration_length - 2; d++) {
                    log_prob = obs + beamlist_duration[t + (d + 1) * beamlist_length];
                    beamlist_x_duration[t + d * x_factor] = log_prob;
                    dur_sum = lse(dur_sum, log_prob);
                }
                log_prob = obs + logf(1 - tail) + beamlist_duration[t + (duration_length - 1) * beamlist_length];
                beamlist_x_duration[t + (duration_length - 2) * x_factor] = log_prob;
                dur_sum = lse(dur_sum, log_prob);
                log_prob = obs + logf(tail) + beamlist_duration[t + (duration_length - 1) * beamlist_length];
                beamlist_x_duration[t + (duration_length - 1) * x_factor] = log_prob;
                dur_sum = lse(dur_sum, log_prob);
                beamlist_x_score[t] = dur_sum;
                merge_id[t] = (parent_x_id[t] << 2) + (kp >> (2 * (kmer_length - 1))); // Unique ID for merge decision
                traceback_x[t] = t;

                // Expand all children
                for (int b = 0; b < 4; b++) {
                    int kt = kp + b * total_kmers; // Kmer transion
                    int kc = kt >> 2; // Child kmer
                    int tt = t + (b + 1) * beamlist_length; // Place in expanded list
                    obs = logf(nn_prior[kc + n * total_kmers]);
                    current_x_id[tt] = tt + (n + 1)* x_factor; // New unique ID for beam
                    parent_x_id[tt] = current_id[t];
                    beamlist_x_kmer[tt] = kc;
                    dur_sum = -flt_large;
                    for (int d = 0; d < duration_length; d++) {
                        log_prob = obs + logf(transition_probability[kt]) + beamlist_duration[t + 0 * beamlist_length] + logf(duration_probability[d]);
                        beamlist_x_duration[tt + d * x_factor] = log_prob;
                        dur_sum = lse(dur_sum, log_prob);
                    }
                    beamlist_x_score[tt] = dur_sum;
                    merge_id[tt] = (parent_x_id[tt] << 2) + b; // Unique ID for merge decision
                    traceback_x[tt] = t - beamlist_length; // Negative index for new base
                }
            }

            // Merge
            sort(merge_id, list_pos, sortlist_length);
            for (int t = threadIdx.x; t < (4 + 1) * beamlist_length; t += blockDim.x) {
                if (merge_id[t] == merge_id[t + 1]) { // Found two elements to merge
                    // parent id and kmer are allready the same
                    int t1 = list_pos[t];
                    int t2 = list_pos[t + 1];
                    if ((parent_x_id[t1] != parent_x_id[t2]) | (beamlist_x_kmer[t1] != beamlist_x_kmer[t2])) {
                        printf("Found an error in the code (batch_id=%d, n=%d, t1=%d, t2=%d, t=%d)!\n",blockIdx.x,n,t1,t2,t);
                        printf("  parent_x_id[t1] = %d)!\n", parent_x_id[t1]);
                        printf("  parent_x_id[t2] = %d)!\n", parent_x_id[t2]);
                        printf("  beamlist_x_kmer[t1] = %d)!\n", beamlist_x_kmer[t1]);
                        printf("  beamlist_x_kmer[t2] = %d)!\n", beamlist_x_kmer[t2]);
                        printf("  merge_id[t] = %d)!\n", merge_id[t]);
                        printf("  merge_id[t + 1] = %d)!\n", merge_id[t + 1]);
                    }

                    for (int d = 0; d < duration_length; d++) {
                        beamlist_x_duration[t1 + d * x_factor] = lse(beamlist_x_duration[t1 + d * x_factor], beamlist_x_duration[t2 + d * x_factor]);
                    }
                    if (beamlist_x_score[t2] > beamlist_score[t1]) {
                        traceback_x[t1] = traceback_x[t2]; // Keep the highest scoring path in traceback
                    }
                    beamlist_x_score[t1] = lse(beamlist_x_score[t1], beamlist_x_score[t2]);
                    beamlist_x_score[t2] = -flt_large; // Null out left over beam
                    if (current_x_id[t1] > current_x_id[t2]) current_x_id[t1] = current_x_id[t2]; // Always keep the lower id
                }
            }
        }

        // Sort beam list to allow for pruning for the beamlist_lenght best beams
        sort(beamlist_x_score, list_pos, (n == 0) ? total_kmers : sortlist_length);
        for (int t = threadIdx.x; t < beamlist_length; t += blockDim.x) {
            int tt = list_pos[t];
            current_id[t] = current_x_id[tt];
            parent_id[t] = parent_x_id[tt];
            for (int d = 0; d < duration_length; d++) {
                beamlist_duration[t + d * beamlist_length] = beamlist_x_duration[tt + d * x_factor];
            }
            beamlist_score[t] = beamlist_x_score[t]; // Not tt because this list is allready sorted
            beamlist_kmer[t + n * beamlist_length] = beamlist_x_kmer[tt];
            traceback[t + n * beamlist_length] = traceback_x[tt];
        }

        // Synchronize after pruning
        __syncthreads();

        // Normalize using shared memory (for numerical stability)
        float thread_sum = -flt_large;
        for (int t = threadIdx.x; t < beamlist_length; t += blockDim.x) {
            thread_sum = lse(thread_sum, beamlist_score[t]);
        }
        sdata[threadIdx.x] = thread_sum;
        listReduce(sdata, threadIdx.x);
        thread_sum = sdata[0];
        for (int t = threadIdx.x; t < beamlist_length; t += blockDim.x) {
            for (int d = 0; d < duration_length; d++) {
                beamlist_duration[t + d * beamlist_length] -= thread_sum;
            }
            beamlist_score[t] -= thread_sum;
        }
    }

    //return;

    // Backtracking
    __syncthreads();
    if (threadIdx.x == 0) {

        // Best beam is always on top due to prior sorting
        int best_beam = 0;

        // Backtrack
        int t = best_beam;
        int len = 0;
        for (int n = block_length - 1; n >= 0; n--) {
            int tn = traceback[t + n * beamlist_length];
            //printf("Trace back at n=%d is %d\n", n, tn);
            if (tn < 0) {
                output[len++] = beamlist_kmer[t + n * beamlist_length]; // Return kmers
                t = tn + beamlist_length; // Restore index
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
            output[n] = output[len - n - 1];
            output[len - n - 1] = tmp;
        }
    }
}


__global__
void dnaseq_viterbi_kernel(const float* __restrict__ nn_prior, int32_t* __restrict__ output, const float* __restrict__ duration_probability, const float* __restrict__ tail_factor, const float* __restrict__ transition_probability,
    float* __restrict__ alpha_buffer, int32_t* __restrict__ traceback,
    int block_length, int kmer_length, int batch_size, int duration_length) {

    static __shared__ float sdata[max_threads_per_block + 16];
    sdata[threadIdx.x + 16] = -flt_large; // Clear list for bondary free reduction

    // Constants
    int batch_item = blockIdx.x;
    int total_kmers = 1 << (2 * kmer_length);
    int kmer_mask = total_kmers - 1;
    float tail = tail_factor[batch_item];

    // Position bufferes appropriately for current batch item
    nn_prior += total_kmers * block_length * batch_item;
    output += block_length * batch_item;
    alpha_buffer += total_kmers * duration_length * block_length * batch_item;
    traceback += total_kmers * duration_length * block_length * batch_item;
    duration_probability += duration_length * batch_item;

    // Compute forward selection step
    for (int n = 0; n < block_length; n++) {
        float alpha_sum = -flt_large;
        for (int k = threadIdx.x; k < total_kmers; k += blockDim.x) {
            float nn_value = nn_prior[n * total_kmers + k];
            for (int d = 0; d < duration_length; d++) {
                int tpos_best = -1;
                float score_best = -flt_large;
                if (n == 0) {
                    score_best = logf(nn_value);
                }
                else { // n > 0
                    for (int k0 = 0; k0 < 4; k0++) {
                        int kt = (k << 2) + k0; // Transition probability entry
                        int kp = kt & kmer_mask; // Prior kmer
                        int tpos = 0 * total_kmers + kp;
                        float score = alpha_buffer[(n - 1) * total_kmers * duration_length + tpos] + logf(transition_probability[kt]) + logf(duration_probability[d]) + logf(nn_value);
                        if (score > score_best) {
                            score_best = score;
                            tpos_best = tpos;
                        }
                    }
                    if (d < duration_length - 2) {
                        int tpos = (d + 1) * total_kmers + k;
                        float score = alpha_buffer[(n - 1) * total_kmers * duration_length + tpos] + logf(nn_value);
                        if (score > score_best) {
                            score_best = score;
                            tpos_best = tpos;
                        }
                    }
                    if (d == duration_length - 2) {
                        int tpos = (d + 1) * total_kmers + k;
                        float score = alpha_buffer[(n - 1) * total_kmers * duration_length + tpos] + logf(1 - tail) + logf(nn_value);
                        if (score > score_best) {
                            score_best = score;
                            tpos_best = tpos;
                        }
                    }
                    if (d == duration_length - 1) {
                        int tpos = d * total_kmers + k;
                        float score = alpha_buffer[(n - 1) * total_kmers * duration_length + tpos] + logf(tail) + logf(nn_value);
                        if (score > score_best) {
                            score_best = score;
                            tpos_best = tpos;
                        }
                    }
                }
                alpha_buffer[n * total_kmers * duration_length + d * total_kmers + k] = score_best;
                traceback[n * total_kmers * duration_length + d * total_kmers + k] = tpos_best;
                alpha_sum = lse(alpha_sum, score_best);
            }
        }
        // Normalize using shared memory (could provide some stability)
        sdata[threadIdx.x] = alpha_sum;
        listReduce(sdata, threadIdx.x);
        alpha_sum = sdata[0];
        for (int k = threadIdx.x; k < total_kmers; k += blockDim.x) {
            for (int d = 0; d < duration_length; d++) {
                alpha_buffer[n * total_kmers * duration_length + d * total_kmers + k] -= alpha_sum;
            }
        }
    }

    // Find the best path at the end of the trellis
    // Use memory space allocated for first column of alpha and traceback
    int n = block_length - 1;
    for (int k = threadIdx.x; k < total_kmers; k += blockDim.x) {
        float score_best = -flt_large;
        int tpos_best;
        for (int d = 0; d < duration_length; d++) {
            int tpos = d * total_kmers + k;
            float score = alpha_buffer[n * total_kmers * duration_length + tpos];
            if (score > score_best) { // Search over durations
                score_best = score;
                tpos_best = tpos;
            }
        }
        alpha_buffer[k] = score_best;
        traceback[k] = tpos_best;
    }
    sort(alpha_buffer, traceback, total_kmers);

    // Backtracking (single thread)
    __syncthreads();
    if (threadIdx.x == 0) {
        int tpos = traceback[0]; // Get best beam from sorted list
        int len = 0;
        for (int n = block_length - 1; n >= 0; n--) {
            if (tpos < total_kmers) { // State with d = 0, tpos = kmer, output valye
                output[len++] = tpos;
            }
            //if (batch_item == 0) printf("tpos value %d at pos n=%d\n", tpos, n);
            tpos = traceback[n * total_kmers * duration_length + tpos]; // Follow traceback path
        }
        // Pad with stop symbols
        for (int n = len; n < block_length; n++) {
            output[n] = -1;
        }
        // Reverse sequence
        for (int n = 0; n < (len >> 1); n++) {
            int32_t tmp = output[n];
            output[n] = output[len - n - 1];
            output[len - n - 1] = tmp;
        }
    }
}

// Wrappers for cpp call

void dnaseq_lfbs_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    int32_t* current_id, int32_t* parent_id, int32_t* parent_x_id, float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
    int32_t* traceback,
    int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length) {

    cudaError_t status;

    int total_beamtails = 1 << (2 * beamtail_length);
    int threads_per_block = total_beamtails;
    int min_grid_size;
    int max_threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, dnaseq_lfbs_kernel, 0, 0);
    if (threads_per_block > max_threads_per_block) threads_per_block = max_threads_per_block;

    
    clock_t begin = clock();
    

    dnaseq_lfbs_kernel <<< batch_size, threads_per_block, 0, stream >>> (nn_prior, output, duration_probability, tail_factor, transition_probability,
        current_id, parent_id, parent_x_id, beamlist_duration, beamlist_x_duration, beamlist_score, beamlist_x_score,
        traceback, block_length, kmer_length, beamtail_length, batch_size, duration_length);
    
    
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    

    status = cudaGetLastError();
    if (status) {
        std::cout << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    }
}


void dnaseq_beam_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    int32_t* current_id, int32_t* current_x_id, int32_t* parent_id, int32_t* parent_x_id,
    int32_t* __restrict__ beamlist_kmer, int32_t* __restrict__ beamlist_x_kmer,
    float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
    int32_t* merge_id, int32_t* list_pos,
    int32_t* traceback, int32_t* traceback_x,
    int block_length, int kmer_length, int beamlist_length, int batch_size, int duration_length) {

    cudaError_t status;

    int sortlist_length = 1;
    for (int k = 5 * beamlist_length - 1; k > 0; k >>= 1) { // Find minimum length for sorting expanded beam list
        sortlist_length <<= 1;
    }
    int threads_per_block = sortlist_length;
    int min_grid_size;
    int max_threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, dnaseq_beam_kernel, 0, 0);
    if (threads_per_block > max_threads_per_block) threads_per_block = max_threads_per_block;


    clock_t begin = clock();

    dnaseq_beam_kernel <<< batch_size, threads_per_block, 0, stream >>> (nn_prior, output, duration_probability, tail_factor, transition_probability,
        current_id, current_x_id, parent_id, parent_x_id,
        beamlist_kmer, beamlist_x_kmer,
        beamlist_duration, beamlist_x_duration, beamlist_score, beamlist_x_score,
        merge_id, list_pos, traceback, traceback_x,
        block_length, kmer_length, beamlist_length, batch_size, duration_length);

    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    status = cudaGetLastError();
    if (status) {
        std::cout << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    }
}

void dnaseq_viterbi_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    float *alpha_buffer, int32_t * traceback,
    int block_length, int kmer_length, int batch_size, int duration_length) {

    cudaError_t status;

    int total_kmers = 1 << (2 * kmer_length);
    int threads_per_block = total_kmers;
    int min_grid_size;
    int max_threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, dnaseq_viterbi_kernel, 0, 0);
    if (threads_per_block > max_threads_per_block) threads_per_block = max_threads_per_block;

    clock_t begin = clock();

    dnaseq_viterbi_kernel <<< batch_size, threads_per_block, 0, stream >>> (nn_prior, output, duration_probability, tail_factor, transition_probability,
        alpha_buffer, traceback,
        block_length, kmer_length, batch_size, duration_length);
    
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    status = cudaGetLastError();
    if (status) {
        std::cout << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    }
}
