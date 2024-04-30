import sys
import numpy as np
import matplotlib.pyplot as plt
import threading
import csv
import mappy as mp
base_dict_op = {0:'A', 1:'C', 2:'G', 3:'T'}
base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
def str2bool(v):
  if v.lower() in ('yes', 'True','true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no','Flase', 'false', 'f', 'n', '0'):
    return False
  else:
    raise ArgumentTypeError('Boolean value expected.')
def argparser():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,add_help=False)
  parser.add_argument('-output',default='output/',type=str,help='where to put output png')
  parser.add_argument('-i','--input',default='',nargs='+',type=str,help='fasta files')
  parser.add_argument('-name',default='meowmeow',type=str,help='name of the output png, default meowmeow')
  parser.add_argument('-ref',default='example_data/P15202_101_wtdbg2_1xracon_medaka_pilonsnpindel.fasta',type=str,help='')
  parser.add_argument('-error_plot',default=False,type=str2bool,help='whether to plot errors, defult False')
  return parser
def parse_arguments(sys_input):
  parser = argparser()
  return parser.parse_args(sys_input)
def plot_hist(y,save,bins,fname=''):
    y_hist = plt.hist(y,bins=bins)
    x_min = np.min(y)
    x_max = np.max(y)
    mode_idx = np.argmax(y_hist[0])
    print('The mode is at range: ['+str(y_hist[1][mode_idx])+', '+str(y_hist[1][mode_idx+1])+'].')
    title = 'mean: '+str(np.mean(y))+' median: '+str(np.median(y))+' for total '+str(len(y))+' reads'
    print(title)
    plt.title(title)
    for i in bins:
        plt.axvline(x=i,color='y')
    plt.axvline(x=x_min,color='b')
    plt.axvline(x=x_max,color='b')
    plt.axvline(x=y_hist[1][mode_idx],color='r')
    plt.axvline(x=y_hist[1][mode_idx+1],color='r')
    x_bin = (bins[-1]-bins[0])/20
    plt.xticks(np.arange(bins[0],bins[-1],x_bin))
    plt.grid()
    if not save:
        plt.draw()
        plt.pause(1)
        input('<Hit Enter To Continue>')
    else:
        fig = plt.gcf()
        fig.set_size_inches(20,12)
        plt.savefig(fname+'.png',dpi=100)
    plt.close()
# merge two dictionary together
def mergeDict(dict1,dict2):
    if len(dict2) ==0:
        return dict1
    elif len(dict1) == 0:
        return dict2
    else:
        dict3 = {**dict2,**dict1} #update dict2 with dict1 value
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = [value , dict2[key]]
    return dict3
# collect insertion or deletion bases into vector respectively
def collections(vector,seq):
    for s in seq:
        vector[base_dict[s]] += 1
# translate the cs string in paf file into error information, cs has details of substitution
def cs_count(diff_seq,cslong):
    mark = 5 # different sequence starts with head 'cs:Z:', and alignment start with 'equal'
    ident = 0
    insert = 0
    delete = 0
    subs = 0
    del_collect = np.zeros(4)
    ins_collect = np.zeros(4)
    sub_fusion = np.zeros((4,4)) # ref to query
    for i in range(6,len(diff_seq)): # check from 6, just in case
        if diff_seq[i] in ':+-*~=':
            if diff_seq[mark]=='=' and cslong: # identical sign for long format [ACCGT]
                ident += (i - mark - 1)
            if diff_seq[mark]==':' and ~cslong: # identical [1024]
                ident += int(diff_seq[mark+1:i])
            elif diff_seq[mark]=='+': # insertion [agct]
                insert += (i - mark - 1)
                collections(ins_collect,diff_seq[mark+1:i])
            elif diff_seq[mark]=='-': # deletion [agct]
                delete += (i - mark - 1)
                collections(del_collect,diff_seq[mark+1:i])
            elif diff_seq[mark]=='*': # substitution, appears by pairs, e.g. *ag*ga*gc*ca*ct...
                subs += 1
                sub_fusion[base_dict[diff_seq[mark+1]],base_dict[diff_seq[mark+2]]] += 1
            elif diff_seq[i]=='~': # intro length and splice signal ???
                print('unknown sign appears, pause for check')
                import pdb;pdb.set_trace()
            mark = i
    # deal with the last difference which not end with symbols
    if diff_seq[mark]=='=' and cslong:
        ident += (len(diff_seq) - mark - 1)
    elif diff_seq[mark]==':' and ~cslong: # identical [1024]
        ident += int(diff_seq[mark+1:])
    elif diff_seq[mark]=='+': # insertion [agct]
        insert += (len(diff_seq) + 1 - mark)
        collections(ins_collect,diff_seq[mark+1:])
    elif diff_seq[mark]=='-': # deletion [agct]
        delete += (len(diff_seq) + 1 - mark)
        collections(del_collect,diff_seq[mark+1:])
    elif diff_seq[mark]=='*': # substitution, appears by pairs, e.g. *ag a is substituted with wrong g
        subs += 1
        sub_fusion[base_dict[diff_seq[mark+1]],base_dict[diff_seq[mark+2]]] += 1
    elif diff_seq[i]=='~': # intro length and splice signal ???
        print('unknown sign appears, pause for check')
        import pdb;pdb.set_trace()
    if ident == 0:
        import pdb;pdb.set_trace()
        print('ERROR:zero identities appear:')
        print(diff_seq)
    align_len = (ident+insert+delete+subs)
    blast = ident/align_len
    insert = insert/align_len
    delete = delete/align_len
    subs = subs/align_len
    error_table = {'identical':ident,'insertion':insert,'deletion':delete,'substitution':subs,'delet_collection':del_collect,'insertion_collection':ins_collect,'substitution_fusion':sub_fusion,'blast':blast}
    return error_table
# load paf file line by line, might merg it into other functions later
def load_paf(paf_fn,locking,plot=False):
    global insert_sum
    global delete_sum
    global subs_sum
    global identities_sum
    global identities_direct_sum
    global subs_fus_sum
    global lengths_sum
    global fail_reads
    global reads_ids
    identities = list()
    inserts = list()
    deletes = list()
    subs = list()
    identities_direct = list()
    subs_fusions = np.zeros((4,4))
    lengths = list()
    paf_csv = {}
    reads_id = {}
    if 'long' in paf_fn:
        cslong = True
    else:
        cslong = False
    paf = open(paf_fn,'r')
    print('working for '+paf_fn+'...')
    line_count = 1
    for lines in paf:
        line = lines.strip().split('\t')
        if len(line)<9:
            continue
        if half_ecoli==1 and int(line[7])>=2500000:
            continue
        elif half_ecoli==2 and int(line[7])<2500000:
            continue
        # logging for read ID
        read_id = line[0]
        read_length = int(line[1])
        if line[-1][5] == '=':
            cslong = True
        elif line[-1][5] == ':':
            cslong = False
        else:
            print("The cs, cg string did not start with match!")
            print(line)
        #    sys.exit(1)
        ident = int(line[9])/int(line[10])
        identities_direct.append(ident)
        lengths.append(int(line[9]))
        #if ident == 1 and not 'catitude' in read_id:
        #    print('100% identities in '+lines)
        if 'cs' in line[-1] or 'cg' in line[-1]:
            if 'cs' in line[-1]: # check if it has cs string
                cs_seq = line[-1]
                # sometimes there is "NNNNNN" in reference
                if 'n' in cs_seq:
                    print('false reference in '+line[5])
                    with locking:
                        fail_reads += 1
                    #print(line)
                    #import pdb;pdb.set_trace()
                    continue
                cg_seq = line[-2]
                error_cg = cg_count(cg_seq)
                error_cs = cs_count(cs_seq,cslong)
                identities.append(error_cs['blast'])
                inserts.append(error_cs['insertion'])
                deletes.append(error_cs['deletion'])
                subs.append(error_cs['substitution'])
                if read_id not in paf_csv.keys():
                    paf_csv[read_id] = []
                paf_csv[read_id].append(line[0])
                paf_csv[read_id].append(line[1])
                paf_csv[read_id].append(line[2])
                paf_csv[read_id].append(line[3])
                paf_csv[read_id].append(error_cs['blast'])
                paf_csv[read_id].append(error_cs['insertion'])
                paf_csv[read_id].append(error_cs['deletion'])
                paf_csv[read_id].append(error_cs['substitution'])
                subs_fusions += error_cs['substitution_fusion']
            elif 'cg' in line[-1]: # normally don't use cg, if opoun useage, remember to add n check
                cg_seq = line[-2]
                error_cg = cg_count(cg_seq)
                inserts.append(error_cg['insertion'])
                deletes.append(error_cg['deletion'])
    if 'long' in paf_fn:
        paf_name = paf_fn[paf_fn.rfind('/')+1:-9]
    else:
        paf_name = paf_fn[paf_fn.rfind('/')+1:-4]
    d_file = csv.writer(open(output_path+paf_name+".csv", "w"))
    for read in paf_csv.keys():
        d_file.writerow(paf_csv[read][:8])
        if len(paf_csv[read])>8:
            d_file.writerow(paf_csv[read][8:])
    with locking:
        reads_ids = mergeDict(reads_ids,reads_id)
        identities_sum += identities
        identities_direct_sum += identities_direct
        subs_fus_sum += subs_fusions
        insert_sum += inserts
        delete_sum += deletes
        subs_sum += subs
        lengths_sum += lengths
def load_paf_onlyident(paf_fn,locking,plot=False):
    global identities_direct_sum
    global reads_ids
    global lengths_sum
    lengths = list()
    identities_direct = list()
    paf_csv = {}
    reads_id = {}
    if 'long' in paf_fn:
        cslong = True
    else:
        cslong = False
    paf = open(paf_fn,'r')
    print('working for '+paf_fn+'...')
    line_count = 1
    for lines in paf:
        line = lines.strip().split('\t')
        if len(line)<9:
            continue
        if half_ecoli==1 and int(line[7])>=2500000:
            continue
        elif half_ecoli==2 and int(line[7])<2500000:
            continue
        # logging for read ID
        read_id = line[0]
        if line[-1][5] == '=':
            cslong = True
        elif line[-1][5] == ':':
            cslong = False
        else:
            print("The cs, cg string did not start with match!")
            print(line)
        #    sys.exit(1)
        ident = int(line[9])/int(line[10])
        identities_direct.append(ident)
        lengths.append(int(line[9]))
        #if ident == 1 and not 'catitude' in read_id:
        #    print('100% identities in '+lines)
    if 'long' in paf_fn:
        paf_name = paf_fn[paf_fn.rfind('/')+1:-9]
    else:
        paf_name = paf_fn[paf_fn.rfind('/')+1:-4]
    with locking:
        reads_ids = mergeDict(reads_ids,reads_id)
        identities_direct_sum += identities_direct
        lengths_sum += lengths
if __name__ == '__main__':
    paf_files = sys.argv[1:-3]
    if_save_other = sys.argv[-3]
    output_path = sys.argv[-2]
    half_ecoli = int(sys.argv[-1])
    threads = list()
    lock = threading.Lock()
    reads_ids = {}
    identities_direct_sum = list()
    identities_sum = list()
    insert_sum = list()
    delete_sum = list()
    subs_sum = list()
    subs_fus_sum = np.zeros((4,4))
    lengths_sum = list()
    fail_reads = 0
    bins = np.arange(0,1.01,0.01)
    if if_save_other=='n' or if_save_other=='no':
        for i in range(len(paf_files)):
            threads.append(threading.Thread(target=load_paf_onlyident,args=(str(paf_files[i]),lock,)))
        for i in range(len(threads)):
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        print('done')
        print('profiled in total '+str(len(identities_direct_sum))+' reads.')
        plot_hist(identities_direct_sum,save=True,bins=bins,fname=output_path+'identities')
        bins_len = np.arange(0,max(lengths_sum),200)
        plot_hist(lengths_sum,save=True,bins=bins_len,fname=output_path+'lengths')
        print('total bases called: '+str(np.sum(lengths_sum)))
    if if_save_other=='y' or if_save_other=='yes':
        for i in range(len(paf_files)):
            threads.append(threading.Thread(target=load_paf,args=(str(paf_files[i]),lock,)))
        for i in range(len(threads)):
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        print('done')
        print('profiled in total '+str(len(identities_direct_sum))+' reads.')
        print('saving identities...')
        plot_hist(identities_sum,save=True,bins=bins,fname=output_path+'identities')
        print('saving insertions...')
        plot_hist(insert_sum,save=True,bins=bins,fname=output_path+'insertions')
        print('saving deletions...')
        plot_hist(delete_sum,save=True,bins=bins,fname=output_path+'deletions')
        print('saving subtitutions...')
        plot_hist(subs_sum,save=True,bins=bins,fname=output_path+'substitutions')
        print('saving matched lengths...')
        bins_len = np.arange(0,max(lengths_sum),200)
        plot_hist(lengths_sum,save=True,bins=bins_len,fname=output_path+'lengths')
        print('total bases called: '+str(np.sum(lengths_sum)))
else:
    reads_ids = {}
    identities_direct_sum = list()
    identities_sum = list()
    insert_sum = list()
    delete_sum = list()
    subs_sum = list()
    subs_fus_sum = np.zeros((4,4))
    lengths_sum = list()
    fail_reads = 0
