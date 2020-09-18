import sys, random, pdb
from common import *
import Utils

data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])

#data_dir = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Data"
#config_fp = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Configs/config_extract_11_features"

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

    print f

    file_DHS_names.append(f.split(".")[0])

    data_DHS = readFileInTable(data_dir + "/DHS_13cells/" + f)

    data_DHS.sort(cmp=comparePeaks)

    for e_key in enhancer_DHS_correlation.keys():

        e = e_key.split("_")

        indices = getOverlappedPeaks(0, len(data_DHS), data_DHS, e)

        if len(indices) > 0:
            signal = sum([float(data_DHS[i][3]) for i in indices]) /len(indices)
        else:
            signal = 0.0

        enhancer_DHS_correlation[e_key].append(signal)

    for p_key in promoter_DHS_correlation.keys():

        p = p_key.split("_")

        indices = getOverlappedPeaks(0, len(data_DHS), data_DHS, p)

        if len(indices) > 0:
            signal = sum([float(data_DHS[i][3]) for i in indices]) /len(indices)
        else:
            signal = 0.0

        promoter_DHS_correlation[p_key].append(signal)

        
writeDataTableAsCSV([['name_E'] + file_DHS_names] + [[e_key] + enhancer_DHS_correlation[e_key] for e_key in enhancer_DHS_correlation.keys()], data_dir + '/Enhancer_Features/enhancer_dhs')
writeDataTableAsCSV([['name_P'] + file_DHS_names] + [[p_key] + promoter_DHS_correlation[p_key] for p_key in promoter_DHS_correlation.keys()], data_dir + '/Promoter_Features/promoter_dhs')
