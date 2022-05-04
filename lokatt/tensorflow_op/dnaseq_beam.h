#pragma once

#include "cuda.h"
#include <stdint.h>

void dnaseq_beamsearch_gpu(cudaStream_t& stream, float* nn_prior, int32_t* output, float* duration_probability, float* tail_factor, float* transition_probability,
	int32_t* current_id, int32_t* parent_id, int32_t* parent_x_id, float* beamlist_duration, float* beamlist_x_duration, float* beamlist_score, float* beamlist_x_score,
	int32_t* traceback,
	int block_length, int kmer_length, int beamtail_length, int batch_size, int duration_length);