from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import sys
import tensorflow_probability as tfp
import h5py
import time
import threading
from tqdm import tqdm
from tensorflow.python.platform import resource_loader

from .utils_dna import prepro_signal,base2num,seg_assembler
from .tensorflow_op.dnaseq_beam_im import dnaseq_beam
from .model_2RES2BI512_resoriginal import Model_RESBi

# parse keyboard inputs
def str2bool(v):
  if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no','Flase', 'false', 'f', 'n', '0'):
    return False
  else:
    raise ArgumentTypeError('Boolean value expected.')
def argparser():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,add_help=False)
  parser.add_argument('-output',default='output/',type=str,help='where to save fasta file')
  parser.add_argument('-tf_weights','--tf_weights_path',default='model/default/my_checkpoint',type=str,help='TF weights path')
  parser.add_argument('-kmer','--K',type = int,default = 5, help='specify kmer in model init, default 5')
  parser.add_argument('-name',default='meow',type=str,help='fasta name') 
#  parser.add_argument('-d','--device',default=0,type=int,help='which gpu to use, default 0')
  parser.add_argument('-batch',type=int,default=30,help='batch size, default 60')
  #parser.add_argument('-max_reads',type=int,default=-1,help='max reads, default all')
  parser.add_argument('-fast5','--fast5_path',default='example_fast5/',type=str,help='input raw reads file')
  parser.add_argument('-ll',type=str2bool,default=True,help='if use Loglogistic duration estimation per read, default True')
  parser.add_argument('-norm_sig',type=str2bool,default=True,help='if normalize signals, default true')
  
  parser.add_argument('-seg_length',type = int,default = 4096, help='length of signals used for one viterbi,default 4096')
  parser.add_argument('-stride',type=int,default=3800,help='stride for signal segments,default 3800, if set to -1 then no seg is used')
  return parser
def parse_arguments(sys_input):
  parser = argparser()
  return parser.parse_args(sys_input)
def get_params(args):
  PARAMS = parse_arguments(args)
  if not PARAMS.output[-1]=='/':
    PARAMS.output += '/'
  return PARAMS
###########################################################
def write_fasta(f,read_id,read_bases):
    f.write('>'+read_id+'\n')
    for i in range(len(read_bases)//80+1):
        f.write(read_bases[i*80:(i+1)*80])
        f.write('\n')
def means_windowing(x,size=4):
  x = np.array(x)
  x_ = np.zeros((x.size-size+1,size))
  for i in range(size-1):
    x_[:,i] = x[i:-size+i+1]
  x_[:,-1] = x[size-1:]
  means = np.median(x_,axis=1)
  return means
def diffs(x):
  x = np.array(x)
  diff = list()
  for i in range(len(x)-1):
    diff.append(np.abs(x[i+1]-x[i]))
  diff = np.array(diff)
  return diff
def estimate_mu(signals,window_size=4,diff_thre=0.4):
  signals = signals.reshape(-1)
  sig_means = means_windowing(signals,window_size)#median 
  differences = diffs(sig_means)
  duration = len(signals)/np.sum(differences>diff_thre)
  if duration > 20:
    duration = 20
  mu = 0.0927*duration+1.1454
  return mu
#
def finish_fasta(output,batch_id,finished_id,f_output,assembler_percent,lock):
  global total_bases
  global base_dict
  global PARAMS
  global bases_dict
  for i in range(output.shape[0]):
    bases = output[i,output[i,:]>=0]
    bases = base2num(bases.tolist(),PARAMS.K,1,1)
    if len(bases)==0:
      continue
    temp_base = bases[0]
    for j in range(1,len(bases)):
      temp_base += bases[j][-1]
    bases_dict[batch_id[i]].append(temp_base)
  for i in range(len(finished_id)):
    read_bases = seg_assembler(bases_dict[finished_id[i]],p=assembler_percent)
    total_bases += len(read_bases)
    read_id = finished_id[i][5:]
    del bases_dict[finished_id[i]]
    with lock:
      f_output.write('>'+read_id+'\n')
      for i in range(len(read_bases)//80+1):
        f_output.write(read_bases[i*80:(i+1)*80])
        f_output.write('\n')

  
##########################################################
def main(args):
  global total_bases
  global base_dict
  global PARAMS
  global bases_dict
  PARAMS=args
  transition_probability = np.load(resource_loader.get_path_to_datafile('transition_5mer_ecoli.npy')).astype(np.float32)
  batch_size = PARAMS.batch
  seg_length = np.int32(PARAMS.seg_length)
  stride = np.int32(PARAMS.stride)
  assembler_percent = stride/seg_length
  duration = np.zeros([batch_size,16],dtype=np.float32)
  tail_factor = np.zeros([batch_size],dtype=np.float32)
  base_per_read = 0
  #max_reads = PARAMS.max_reads
  max_reads = -1
  batch_duration = np.zeros((batch_size,16)).astype(np.float32)
  batch_tail = np.zeros(batch_size).astype(np.float32)
  batch_input = np.zeros((batch_size,PARAMS.seg_length,1)).astype(np.float32)-100
  batch_id = []
  finished_id = []
  bases_dict = {}
  total_bases = 0
  lock = threading.Lock()
  threads = []
  for i in range(10):
    threads.append(threading.Thread())
  for thre in threads:
    thre.start()
    out_fn = PARAMS.name+'.fasta'
    out_fn = PARAMS.output+out_fn
    f_output = open(out_fn,'w')
  for one_fast5 in os.listdir(PARAMS.fast5_path):
    f = h5py.File(PARAMS.fast5_path+one_fast5,'r')
    if 'unet' in PARAMS.tf_weights_path:
      myNN = Model_UNET(kmer=PARAMS.K)
    else:
      myNN = Model_RESBi(kmer=PARAMS.K)
    test = myNN(np.random.rand(1,4096,1).astype(np.float32))
    myNN.load_weights(PARAMS.tf_weights_path)
    tf_weights_name = PARAMS.name
    print('loaded tf weights at'+PARAMS.tf_weights_path)
    count = 0
    nodes = iter(f.items())
    pbar = tqdm(total=max_reads)
    start = time.time()
    l_duration = np.zeros(16,dtype=np.float32)
    l_tail = np.zeros(1,dtype=np.float32)
    ll=tfp.distributions.LogLogistic(2.1,0.415)
    ll_prob = ll.prob(np.arange(1,20).astype(np.float32))
    l_duration[:15] = ll_prob[:15]
    l_tail[:] = ll_prob[16]/ll_prob[15]
    l_duration[-1] = ll_prob[15]/(1-l_tail)
    cpu_t = threading.Thread()
    cpu_t.start()
    tf_trans = tf.constant(transition_probability)
    for node_name, node in f.items():
      pbar.update(1)
      bases_dict[node_name] = []
      norm_signals = prepro_signal(node['Raw']['Signal'][:]) # no quantized signal
      if PARAMS.ll:
        nowt = time.time()
        if len(norm_signals)>4000:
          mu = estimate_mu(norm_signals)
        else:
          mu = 2.1
        ll=tfp.distributions.LogLogistic(mu,0.415)
        ll_prob = ll.prob(np.arange(1,20).astype(np.float32))
        l_duration[:15] = ll_prob[:15]
        l_tail[:] = ll_prob[16]/ll_prob[15]
        l_duration[-1] = ll_prob[15]/(1-l_tail)
        final_n = int(norm_signals.shape[0]//stride)+1 # we take whatever is at last
      for seg_i in range(final_n):
        batch_id.append(node_name)
        batch_seg_i = len(batch_id)-1
        batch_duration[batch_seg_i,:] = l_duration[:]
        batch_tail[batch_seg_i] = l_tail[:]
        if seg_i*stride+seg_length > norm_signals.shape[0]:
          sig_seg_len = norm_signals.shape[0]-seg_i*stride
          batch_input[batch_seg_i,:sig_seg_len,0] = norm_signals[seg_i*stride:]# whatever is at last
        else:
          batch_input[batch_seg_i,:,0] = norm_signals[seg_i*stride:seg_i*stride+seg_length]
        if len(batch_id)==batch_size:
          Px = myNN(batch_input)
          output = dnaseq_beam(Px+1e-5, tf.Variable(batch_duration), tf.Variable(batch_tail),tf_trans)
          for thre_i in range(len(threads)):
            if not threads[thre_i].is_alive():
              threads[thre_i] = threading.Thread(target=finish_fasta,args=(output.numpy(),batch_id[:],finished_id[:],f_output,assembler_percent,lock))
              threads[thre_i].start()
              break
          alive_T = 0
          for thre in threads:
            if thre.is_alive():
              alive_T += 1
          count += len(finished_id)
          if count%100==0:
            tqdm.write('basecalling speed '+str(total_bases/(time.time()-start))+' bases/sec')
            total_bases = 0
            start=time.time()
          finished_id = []
          batch_id = []
          batch_input[:] = -100
      finished_id.append(node_name)
