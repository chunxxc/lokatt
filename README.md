
<img align="up" src="https://github.com/chunxxc/lokatt/blob/main/lokatt_paint.jpg" alt="drawing" width="400"> 

# Lokatt
An open source HMM-DNN nanopore DNA basecaller

## Installation
### Install from source (recommended)
It is recommemded to install the lokatt basecaller from source. This only requires the tensorflow\_op/dnaseq\_beam.so itself to be compiled beforehand.

Inside this git directory:
```bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o ./lokatt/tensorflow_op/dnaseq_beam.cu.o ./lokatt/tensorflow_op/dnaseq_beam.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3 --use_fast_math -maxrregcount 32

g++  -shared ./lokatt/tensorflow_op/dnaseq_beam.cc ./lokatt/tensorflow_op/dnaseq_beam.cu.o -o ./lokatt/tensorflow_op/dnaseq_beam.so -fPIC -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

pip install -r requirements.txt

pip install ./

```
### Install through pip package
You can install Lokatt as a pre-compiled wheel file. Currently the basecaller only supports Tensorflow>2.8.0 and GPU with CUDA 12.2.

```bash
  python -m pip install ./dist/lokatt-0.1.0-py3-none-any.whl
```
However, after installing the package, you may still need to recompile tensorflow\_op/dnaseq\_beam.so to match your local CUDA. You will need to first locate the directory that lokatt was installed in, then run the first four commands from "Install from source". 
## Usage
### Basecalling
Use the following command to basecall fast5 files inside example\_fast5/ and generate output/meow.fasta with default Lokatt model
```bash
  lokatt basecaller -fast5 example_fast5/ -output output/ -batch 60 -name meow
```
## Repository Overview
A quick overview of the folder structure of this template repository.
```bash
.
├── dist
│   ├── lokatt-0.1.0-py3-none-any.whl
│   ├── lokatt-0.1.0.tar.gz
├── example_fast5
│   └── FAR64318_97d55db5_97.fast5
│   └── Ecoli_illumina_contig.fasta
├── LICENSE
├── lokatt
│   ├── __init__.py
│   └── utils_dna.py
│   ├── error_summary.py
│   ├── gpu_beamsearch_opt.py
│   ├── transition_5mer_ecoli.npy
│   ├── model_2RES2BI512_resoriginal.py
│   └── tensorflow_op
│       ├── dnaseq_beam.cc
│       ├── dnaseq_beam.cu
│       ├── dnaseq_beam.cu.o
│       ├── dnaseq_beam.h
│       ├── dnaseq_beam_im.py
│       ├── dnaseq_beam.o
│       ├── dnaseq_beam.so
│       └── __init__.py
├── makefile
├── MANIFEST.in
├── model
│   └── default
│       ├── checkpoint
│       ├── my_checkpoint.data-00000-of-00001
│       └── my_checkpoint.index
├── output
│   ├── identities.png
│   ├── lengths.png
│   ├── moew.fasta
│   └── moew_nosec.paf
├── README.md
├── requirements.txt
└── setup.py
```
