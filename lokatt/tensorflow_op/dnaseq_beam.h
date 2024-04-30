#pragma once

#include "cuda.h"
#include <stdint.h>

void dnaseq_lfbs_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
	int32_t* current_id, int32_t* parent_id, int32_t* parent_x_id, float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
	int32_t* traceback,
	int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length);

void dnaseq_beam_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    int32_t* current_id, int32_t* current_x_id, int32_t* parent_id, int32_t* parent_x_id,
    int32_t* __restrict__ beamlist_kmer, int32_t* __restrict__ beamlist_x_kmer,
    float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
    int32_t* merge_id, int32_t* list_pos,
    int32_t* traceback, int32_t* traceback_x,
    int block_length, int kmer_length, int beamlist_length, int batch_size, int duration_length);

void dnaseq_viterbi_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
    float* alpha_buffer, int32_t* traceback,
    int block_length, int kmer_length, int batch_size, int duration_length);