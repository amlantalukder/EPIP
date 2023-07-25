import sys
import random
import pdb
from common import *
import Utils

printDec('CalculateDHSOverlap Start')

data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])

# data_dir = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Data"
# config_fp = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Configs/config_extract_11_features"

configs = Utils.read_config_file(config_fp)

enhancers_path = configs["Enhancers_path"]
promoters_path = configs["Promoters_path"]

enhancers = readFileInTable(data_dir + "/" + enhancers_path + "/enhancers")
promoters = readFileInTable(data_dir + "/" + promoters_path + "/promoters")

enhancer_DHS_correlation = dict([["_".join(e[:3]), []] for e in enhancers])
promoter_DHS_correlation = dict([["_".join(p[:3]), []] for p in promoters])

file_DHS_names = []

files_DHS = os.listdir(data_dir + "/DHS_13cells")

for f in files_DHS:

    if not os.path.isfile(data_dir + "/DHS_13cells/" + f):
        continue

    print(f)

    file_DHS_names.append(f.split(".")[0])

    data_DHS = readFileInTable(data_dir + "/DHS_13cells/" + f)

    data_DHS = sorted(data_DHS, key=cmp_to_key(comparePeaks))

    enhancers_ = [e_key.split('_') for e_key in enhancer_DHS_correlation]
    overlapped_indices = getOverlappedPeaks2(enhancers_, data_DHS)

    for i, e in enumerate(enhancers_):

        if i in overlapped_indices:
            signal = sum([float(data_DHS[j][3]) for j in overlapped_indices[i]]) / len(overlapped_indices[i])
        else:
            signal = 0.0

        enhancer_DHS_correlation['_'.join(e)].append(signal)

    promoters_ = [p_key.split('_') for p_key in promoter_DHS_correlation]
    overlapped_indices = getOverlappedPeaks2(promoters_, data_DHS)

    for i, p in enumerate(promoters_):

        if i in overlapped_indices:
            signal = sum([float(data_DHS[j][3]) for j in overlapped_indices]) / len(overlapped_indices)
        else:
            signal = 0.0

        promoter_DHS_correlation['_'.join(p)].append(signal)

writeDataTableAsCSV([['name_E'] + file_DHS_names] + [[e_key] + enhancer_DHS_correlation[e_key] for e_key in enhancer_DHS_correlation.keys()], data_dir + '/Enhancer_Features/enhancer_dhs')
writeDataTableAsCSV([['name_P'] + file_DHS_names] + [[p_key] + promoter_DHS_correlation[p_key] for p_key in promoter_DHS_correlation.keys()], data_dir + '/Promoter_Features/promoter_dhs')

printDec('CalculateDHSOverlap End')
