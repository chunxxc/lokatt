
<img align="right" src="https://github.com/chunxxc/lokatt/blob/main/lokatt_paint.jpg" alt="drawing" width="400"> 

# lokatt
An open source HMM-DNN nanopore DNA basecaller

## Installation
### Install through pip package
You can install Lokatt as a pre-compiled wheel file. Currently the basecaller only supports Tensorflow 2.8.0 and GPU with CUDA 11+.

Inside lokatt/ directory:
```bash
  pip install ./dist/lokattcu110-0.0.1-py3-none-any.whl
  pip install ./dist/lokattcu112-0.0.1-py3-none-any.whl
```
### Install from source
You can compile the lokatt tensorflow operation from source. This requires the tensorflow itself to be compiled from source or running within a docker image.

Inside lokatt/ dorectory:
```bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o ./lokatt/tensorflow_op/dnaseq_beam.cu.o ./lokatt/tensorflow_op/dnaseq_beam.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O3 --use_fast_math -maxrregcount 32

g++  -shared ./lokatt/tensorflow_op/dnaseq_beam.cc ./lokatt/tensorflow_op/dnaseq_beam.cu.o -o ./lokatt/tensorflow_op/dnaseq_beam.so -fPIC -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -lcudart ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

pip3 install -r requirements.txt

python3 setup.py install

```
## Usage
### Basecalling
Use the following command to basecall fast5 files inside example\_fast5/ and generate output/meow.fasta with default Lokatt model
```bash
  lokatt basecaller -fast5 example_fast5/ -output output/ -batch 60 -name meow
```
### Evaluation
## Repository Overview
A quick overview of the folder structure of this template repository.
```bash
.
├── dist
│   ├── lokatt-0.0.1-py3-none-any.whl
│   ├── lokatt-0.0.1.tar.gz
│   ├── lokattcu110-0.0.1-py3-none-any.whl
│   └── lokattcu112-0.0.1-py3-none-any.whl
├── example_fast5
│   └── FAR64318_97d55db5_97.fast5
├── LICENSE
├── lokatt
│   ├── error_summary.py
│   ├── gpu_beamsearch_opt.py
│   ├── __init__.py
│   ├── model_2RES2BI512_resoriginal.py
│   ├── tensorflow_op
│   │   ├── dnaseq_beam.cc
│   │   ├── dnaseq_beam.cu
│   │   ├── dnaseq_beam.cu.o
│   │   ├── dnaseq_beam_DNA-NN.so
│   │   ├── dnaseq_beam.h
│   │   ├── dnaseq_beam_im.py
│   │   ├── dnaseq_beam.o
│   │   ├── dnaseq_beam.so
│   │   ├── dnaseq_beam_tf_cu112docker.so
│   │   ├── __init__.py
│   ├── transition_5mer_ecoli.npy
│   └── utils_dna.py
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
│   ├── lokattoutput.fasta
│   └── lokattoutput_nosec.paf
├── README.md
├── requirements.txt
└── setup.py
```
