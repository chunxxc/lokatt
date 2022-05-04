# in this script, K is kmer
import sys
import os
import numpy as np
from difflib import SequenceMatcher as matcher
# import tables # do not involve if not needed
np.set_printoptions(threshold=sys.maxsize)
base_dict_op = {0:'A', 1:'C', 2:'G', 3:'T'}
base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
def pre_kmer(kmer,k,pre_k):
  K = 4**k
  kpow = K/4
  for i in range(4):
    pre_k[i] = i
  kmer = kmer%(kpow)*4
  for i in range(4):
    pre_k[i] = pre_k[i] + kmer

def complement_DNA(seq):
    if type(seq)==str:
        new_seq = ''
        for i in range(len(seq)):
            if seq[i]=='A' or seq[i]=='a':
                new_seq+='T'
            elif seq[i]=='C' or seq[i]=='c':
                new_seq+='G'
            elif seq[i]=='G' or seq[i]=='g':
                new_seq+='C'
            elif seq[i]=='T' or seq[i]=='t':
                new_seq+='A'
    else:
        new_seq = np.zeros(len(seq))
        new_seq[seq==0]=3
        new_seq[seq==1]=2
        new_seq[seq==2]=1
        new_seq[seq==3]=0
    return new_seq[::-1]
def seg_assembler(base_list,p=0.5,bia=0.25):
    base_list = [base for base in base_list if base is not None]
    T = len(base_list)
    ass = ''
    for i in range(T-1):
        matches = matcher(a=base_list[i],b=base_list[i+1],autojunk=False)
        len1 = len(base_list[i])
        len2 = len(base_list[i+1])
        match_idx = matches.find_longest_match(int(np.floor(len1*(p-bia))),len1,0,int(np.ceil(len2*(1-p+bia))))
        if i==0:
            if match_idx[2]==0:
                ass = ass + base_list[i]+base_list[i+1]
            else:
                ass = base_list[i][:match_idx[0]] + base_list[i+1][match_idx[1]:]
        else:
            if match_idx[2]==0:
                ass += base_list[i+1]
            else:
                ass = ass[:-len(base_list[i])+match_idx[0]] + base_list[i+1][match_idx[1]:]
    return ass
def translate_result(result_dict,k):
    bases = []
    max_len = 0
    for i in range(len(result_dict)):
        kmer_n = result_dict[i]
        kmer_n = kmer_n[kmer_n>=0]
        kmer_1 = base2num(kmer_n[0],k,op=1,flip=1)[0]
        kmer_inv = base2num(kmer_n[1:],1,op=1,flip=1)
        kmer_all = base2kmer(kmer_1+kmer_inv,k,1)
        if len(kmer_all)>max_len:
            max_len = len(kmer_all)
        bases.append(kmer_all)
    pad_bases = np.ones((len(bases),max_len),dtype=np.int32)*-1
    for i in range(len(bases)):
        pad_bases[i,:len(bases[i])] = bases[i]
    return pad_bases
def translate_result_viterbi(result_dict,k):
    bases = []
    for i in range(len(result_dict)):
        kmer_n = result_dict[i]
        kmer_n = kmer_n[kmer_n>=0]
        kmer_1 = base2num(kmer_n[0],k,op=1,flip=1)[0]
        kmer_inv = base2num(kmer_n[1:],k,op=1,flip=1)
        bases.append(kmer_1+''.join(kmer[-1] for kmer in kmer_inv))
    return bases
def MAD_normalize(x):
    x_m = np.median(x)
    x_mad = np.median(np.abs(x-x_m))
    return (x-x_m)/x_mad
def best_z_normalize(x):
  x_norm = np.zeros(len(x))
  wind = len(x)#int(len(x)/2)
  for i in range(len(x)//wind):
    xx = x[i*wind:(i+1)*wind].copy()
    x_mean = np.mean(xx)
    x_std = np.std(xx)
    xx[xx>(x_mean+3*x_std)] = x_mean+3*x_std
    xx[xx<(x_mean-3*x_std)] = x_mean-3*x_std
    x_norm[i*wind:(i+1)*wind] = (xx - x_mean)/x_std
  if len(x)%wind>300:
    xx = x[(i+1)*wind:]
    x_mean = np.mean(xx)
    x_std = np.std(xx)
    xx[xx>(x_mean+3*x_std)] = x_mean+3*x_std
    xx[xx<(x_mean-3*x_std)] = x_mean-3*x_std
    x_norm[i*wind:(i+1)*wind] = (xx - x_mean)/x_std
  else:
    x_norm[i*wind:(i+1)*wind] = (xx - x_mean)/x_std
  return x_norm
def quantize(x,high=5,low=-5,lvls=101):
    '''
    x_q = (x - low)//((high-low)/lvls)
    x_q[x_q>lvls] = lvls
    x_q[x_q<0] = 0
    '''
    x[x<low] = low
    x[x>high] = high
    x_q = np.round((lvls-1)*(x-low)/(high-low))
    return x_q.astype(np.int32)
def prepro_signal(x,mod='z'):
    if mod == 'MAD':
        x = MAD_normalize(x)
    else:
        x = best_z_normalize(x)
    #x_q = quantize(x,5,-5,101)
    return x
def A_init(K):#inital A with 0.25
  base_dict_op = {0:'A', 1:'C', 2:'G', 3:'T'}
  A = np.zeros((4**K,4**K))
  for i in range(4**K):
    base = base2num([i],K,op=True)[0]
    for j in range(4):
      indx = base2num([base[-K+1:]+base_dict_op[j]],K)
      A[i,indx] = 0.25
  return A
# calculate prior of X based on MLE
def prior_init(Z,K,mod='unit'):
  Po = np.ones(4**K)/4**K
  if mod=='unit':
    return Po
  if mod == 'random':
    Po = np.random.rand(4**K)
    Po = Po/Po.sum()
    return Po
  else:
    Z = Z.astype(int)
    T = len(Z)
    for t in range(T):
      Po[Z[t]] = Po[Z[t]]+1
    #import pdb; pdb.set_trace()
    row_sums = Po.sum()
    Po = Po/row_sums
    #print('the prior by MLE is:'+"\n"+str(Po))
    print('done')
    return Po
# calculate transition table based on MLE
def Pxz_init(K,bin_num=0):
  bin_num = int(bin_num)
  if bin_num != 0:
    Pxz = np.random.rand(4**K,bin_num)
    Pxz = np.divide(Pxz,np.sum(Pxz,axis=1).reshape(-1,1))
  return Pxz
def transition_compute(Z_,K):
  # Z_ better be output of base2kmer
  print('caluating transition matrix by MLE...')
  Z_ = Z_.astype(int)
  T = len(Z_)
  #A = A_init(K)
  A = np.zeros((4**K,4**K))
  Zb = -1
  for t in range(T):
    Za = Z_[t]
    if Zb >= 0:
      A[Zb,Za] = A[Zb,Za]+1
      Zb = Za
    else:
      Zb = Za
      continue
    #if Za == 4**K-1 and Zb<252:
    #  print(Z)
    #  print(t)
  #import pdb; pdb.set_trace()
  #A = A/A.sum(axis=1).reshape(-1,1)
  #A = A + np.diag(np.ones(4**K)*4/5)
  #print('the transition matrix by MLE is:'+"\n"+str(A))
  print('done')
  return A
###############################################################################################
def normalize(x,op='standarlise'):
  if op=='normalize':
    x_min = np.min(x)
    x_max = np.max(x)
    y = (x-x_min)/(x_max-x_min)
  if op=='standarlise':
    x_bar = np.mean(x)
    x_var = np.var(x)
    y = (x-x_bar)/np.sqrt(x_var)
  if op == 'median_win':
    med = np.median(x)
    MAD = np.median(np.absolute(x-med))
    x[x>med+5*MAD]=med+5*MAD
    x[x<med-5*MAD]=med-5*MAD
    y=(x-med)/MAD
  return y
def base2table(bases,k=4):
  if type(bases[0])==str:
    # from ACTG to one-hot-coding table of 0~4^k
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    num_kmer = len(bases)
    output_idx = np.zeros((num_kmer,4**k))
    output_basenum = np.zeros(num_kmer)
    for j in range(num_kmer):
      kmer = bases[j]
      #print(kmer)
      indx = 0
      for i in range(k-1,-1,-1):
        indx += base_dict[kmer[k-1-i]]*4**i
      output_idx[j,int(indx)] = 1
      output_basenum[j] = indx
    return output_idx.astype(int), output_basenum  
  else:
    num_kmer = np.shape(bases)[0]
    bases = bases.reshape(num_kmer)
    output_idx = np.zeros((num_kmer,4**k))
    for j in range(num_kmer):
      output_idx[j,bases[j]] = 1
    return output_idx.astype(int)
def base2vec(bases,k=4):
  # from ACTG to one-hot-coding of 0~4*k
  base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
  num_kmer = len(bases)
  output_vec =np.zeros((num_kmer,k*4))
  for i in range(num_kmer):
    for j in range(k):
      output_vec[i,j*4+base_dict[bases[i][j]]] = int(1)
  #output_vec = int(output_vec)
  return output_vec.astype(int)
def base2kmer(Z,K,flip=True,op=False):
    if not op:
        T = len(Z)
        Z_kmer = np.zeros(T-K+1)
        for i in range(K,T+1):
            Z_kmer[i-K]=base2num(Z[i-K:i],K,flip=flip)[0]
        return Z_kmer.astype(np.int)
    else:
        try:
            Z=Z.tolist()
        except:
            pass
        bases = base2num(Z,K,1,1)
        fbase = bases[0]
        for i in range(1,len(bases)):
            fbase+=bases[i][-1]
        return fbase
def extract_mid_base(Z,K):
  mid_k = int(K//2)
  mid_base = np.zeros(Z.shape)-1
  for j in range(K-1,mid_k,-1):
    Z = Z%(4**j)
  mid_base = Z//(4**(j-1))
  return mid_base.astype(int)
def base2num(Z, K, op=False,flip=False):
  #import pdb;pdb.set_trace()
  # for transfer base to indx
  if not op:
    #print('\n transfering base to indx...\n ')
    if type(Z) != list:
      if type(Z) == np.ndarray:
        Z = Z.tolist()
      else:
        Z = list([Z])
    T = len(Z)
    Z_ = np.zeros(T)  
    for i in range(T):
      kmer = Z[i]
      indx = 0
      for j in range(K-1,-1,-1):
        if flip:
          indx += base_dict[kmer[j]]*(4**j)
        else:
          indx += base_dict[kmer[K-1-j]]*(4**j)
      Z_[i] = indx
    Z_ = Z_.astype(int)
  else:
    #print('\n transfering indx to base...\n ')
    if type(Z) != list:
      if type(Z) == np.ndarray:
        Z = Z.tolist()
      else:
        Z = list([Z])
    T = len(Z)
    Z_ = list()
    for i in range(T):
      kmer = Z[i]
      base = None
      for j in range(K-1,-1,-1):
        indx = kmer//(4**j)
        kmer = kmer%(4**j)
        if not base:
          base = base_dict_op[indx]
          continue
        base += base_dict_op[indx] 
      Z_.append(base)
    if flip:
      for i in range(len(Z_)):
        Z_[i] = Z_[i][::-1]
    if K == 1:
      Z_ = ''.join(map(str,Z_))
    else:
      Z_ = np.asarray(Z_)
    #import pdb;pdb.set_trace()
  return Z_
def transferflip(base_i,K):
  flip_i = base2num(base2num(base_i,K,op=True),K,flip=True)
  return flip_i
def mid_kmer(k1,k2):
    k_s = int((k1-k2)//2)
    kmer1_list = base2num(np.arange(4**k1),k1,1,1)
    kmer12_list = [bases[k_s:k1-k_s] for bases in kmer1_list]
    kmer12 = base2num(kmer12_list,k2,0,1)
    return kmer12
################################################################################################
def assembler(Z_idx,Z_base,kmer,start=0):
  repeat_idx = np.zeros(4)
  for i in range(4):
    for j in range(kmer):
      repeat_idx[i] += i*(4**j)
  repeat_base = base2num(repeat_idx,kmer,op=True) # only care about 'AAA' 'CCC' 'GGG' 'TTT'
  #print(repeat_test)
  T = len(Z_idx)
  base = ''
  repeat = 0
  for i in range(T):
    #isspecial = len(np.where(repeat_test==Z_idx[i])[0])>0
    if len(base)==0:
      base = base + Z_base[i][start:]
      '''
    elif Z_idx[i-1]==Z_idx[i]:# only care about 'AAA' 'CCC' 'GGG' 'TTT'
      if Z_base[i] in repeat_base:
        repeat = repeat + 1
        if repeat > repeat_len:
          base = base + Z_base[i][-1]
          repeat = 0
      else:
        continue
      '''
    elif i==T-1:
      base = base + Z_base[i][start:]
    else:
      repeat = 0
      base = base + Z_base[i][start]
  return base #string
################################################################################################
if __name__ == '__main__':
  # should print [57 39 63]
  base_test = 'TGCGCTTTT'
