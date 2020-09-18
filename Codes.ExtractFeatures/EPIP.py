import sys, random, pdb, os

curr_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

from shutil import copyfile
from common import *
import pandas as pd
import argparse, getopt
from sklearn.externals import joblib
sys.path.insert(0, par_dir + '/Codes.TrainTestModels')
import voting

def getPromoters(gene_file, chrom_size_file):

    genes = readFileInTable(gene_file)

    tss = [[item[0]] + ([item[1], item[1]] if item[5] == "+" else [item[2], item[2]]) + [item[3], item[4], item[5]] for item in genes]
    
    writeDataTableAsText(tss, DATA_FOLDER + "/Promoters/tss.bed")

    os.system("bedtools slop -s -i " + DATA_FOLDER + "/Promoters/tss.bed -g " + chrom_size_file + " -l 1000 -r 100 > " + DATA_FOLDER + "/Promoters/promoters_with_genes.bed")

    promoters = readFileInTable(DATA_FOLDER + "/Promoters/promoters_with_genes.bed")

    promoters = [item[:3] + ["_".join(item[:3])] + item[4:6] for item in promoters]

    writeDataTableAsText(promoters, DATA_FOLDER + "/Promoters/promoters")

    
def generateEPPairs(enhancers, promoters, distance):

    print "Gathering all enhancer promoter pairs within 2Mbp distance"
        
    test_data_set = []

    perc = 10
    j = 0
    
    for e in enhancers:

        left_boundary = int(e[1]) - 2000000
        right_boundary = int(e[2]) + 2000000

        start = 0
        end = len(promoters)

        relevant_promoters = [] 

        while end > start:

            mid = (start + end)/2

            if promoters[mid][0] > e[0]:
                end = mid
            elif promoters[mid][0] < e[0]:
                start = mid + 1
            elif int(promoters[mid][1]) > right_boundary:
                end = mid
            elif int(promoters[mid][2]) < left_boundary:
                start = mid + 1
            else:

                for k in range(mid+1, end):
                    if e[0] == promoters[k][0] and overlapped(left_boundary, right_boundary, promoters[k][1], promoters[k][2]):
                        relevant_promoters.append(promoters[k])
                    else:
                        break

                for k in range(mid, start-1, -1):
                    if e[0] == promoters[k][0] and overlapped(left_boundary, right_boundary, promoters[k][1], promoters[k][2]):
                        relevant_promoters.append(promoters[k])
                    else:
                        break

                break

        e_id = "_".join(e[:3])
        
        test_data_set += [(e_id, "_".join(p[:3]), "1") for p in relevant_promoters]

        j += 1

        perc = showPerc(j, len(enhancers), perc)

    return list(set(test_data_set))

def showError(msg):
    print('----------- Error !!! ------------')
    print(msg)
    print('----------------------------------')

def showArgError(parser):
    parser.parse_args(['-h'])

def combinePredictions(y_pred1, y_pred2):

    y_pred = []

    for i in range(len(y_pred1)):

        # EPIP predicts a pair negative, only when both balanced and unbalanced models predict it as "negative"
        # otherwise EPIP predicts that pair as "positive".
        if y_pred1[i] == y_pred2[i] and y_pred1[i] == 0:
            y_pred.append(0)
        else:
            y_pred.append(1)

    return y_pred

def predict(model_file_balanced, model_file_unbalanced):

    print(model_file_balanced, model_file_unbalanced)

    #--------- Load model ---------
    model_balanced = joblib.load("%s/%s" % (MODEL_DIR, model_file_balanced))
    model_unbalanced = joblib.load("%s/%s" % (MODEL_DIR, model_file_unbalanced))

    outdir = "%s/prediction_results" % (RESULTS_FOLDER)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(len(cell_lines)):

        print cell_lines[i]

        #--------- Gather the specified test data ---------
        test_data_with_features = readFileInTable("%s/TempFeats/%s_temp_feats/features_mean_signals_EP_%s.csv" % (DATA_FOLDER, cell_lines[i], cell_lines[i]), ",")

        #--------- Separate features from the test data set to train the model ---------
        print "Separating test data features for prediction..."
        names_test = []
        x_test = []
        perc = 10

        headers = test_data_with_features[0]
        
        for j in range(1, len(test_data_with_features)):

            names_test.append(test_data_with_features[j][0])
            x_test.append(test_data_with_features[j][1:-1])
            
            perc = showPerc(j, len(test_data_with_features), perc)

        if len(x_test) == 0:
            print "No test data..."
            continue

        print "Convert the features to dataframe..."
        x_test = pd.DataFrame(data=x_test, columns=headers[1:-1])

        #--------- Predict --------
        print "Predicting test data of size", len(x_test)
        y_pred1 = model_balanced.predict(x_test)
        y_pred2 = model_unbalanced.predict(x_test)

        y_pred = combinePredictions(y_pred1, y_pred2)

        output = []

        for j in range(len(names_test)):
            if y_pred[j] == 1:
                [eid, pid] = names_test[j].split("-")
                gene = promoters_gene_dict[pid]

                output.append([eid, gene])

        output.sort(key=lambda x:(x[0], x[1]))

        #--------- Save the prediction results ----------
        writeDataTableAsText(output, "%s/%s" % (outdir, cell_lines[i]))

DATA_FOLDER = par_dir + "/Data"
RESULTS_FOLDER = par_dir + "/Results"
CONFIG_FP = par_dir + "/Configs/config"
CONFIG_FP_CUSTOMIZED = par_dir + "/Configs/config_temp"
MODEL_DIR = par_dir + "/TrainedModels"
ALLOW_WINDOW_FEATURES = "0"

configs = dict(readFileInTable(CONFIG_FP, ":"))
cells_default = configs['Cells']
features = configs['Features']

enhancer_path_system = configs['Enhancers_path']
promoter_path_system = configs['Promoters_path']
ep_pairs_path_system = configs['Pairs_path']

parser = argparse.ArgumentParser(description='Find enhancer targets', epilog='Example: python EPIP.py -e test_enhancers.bed -g test_genes.bed')
required_arg = parser.add_argument_group('required arguments')
required_arg.add_argument('-e', help="Path for hg19 enhancers in bed file format", required=True, metavar="ENHANCER_FILE_PATH")
parser.add_argument('-c', default="None", help='Comma delimited cell lines among {' + cells_default + '}.' + \
                    'In order to predict for a cell line that is not in the list above, user must provide the file path of the features to use for that cell line with "-f" command', metavar="CELL_LINE")
parser.add_argument('-g', default="None", help='Path for human genes in bed file format (default: Use human Gencode 19)', metavar="GENE_FILE_PATH")
parser.add_argument('-s', default="None", help='Path for chromosome sizes file of human genome in tab delimited file format (default: Use human Gencode 19 chrom sizes)', metavar="CHROM_SIZES_FILE_PATH")
parser.add_argument('-d', type=int, default="2000000", help='Maximum distance between enhancers and promoters (default: 2 Mbp)', metavar="INTEGER")
parser.add_argument('-f', default="None", help='Path of the directory where the feature peak files for each cell line are specified in the following formats ' + \
                    'feature_<cellline>_<feature_name>_peaks.[bed|narrowPeak|broadPeak]. <feature_name> must be among the following features.\n' + \
                    '{' + features + '}', metavar="FEATURE_DIR_PATH")

if '-h' in sys.argv[1:]:
    showArgError(parser)
    exit()

opts,args=getopt.getopt(sys.argv[1:],'e:g:s:c:d:f:')

enhancer_path = ""
cells = cells_default
gene_path = DATA_FOLDER + "/Promoters/genes_hg19_gencode.bed"
genome_chrom_sizes_path = DATA_FOLDER + '/Promoters/hg19.chrom.sizes'
distance = 2000000
feature_dir = "None"
epilog='Example of use'
for i in opts:
    if i[0] == '-e':
        enhancer_path = i[1]
    if i[0] == '-g':
        gene_path = i[1]
    if i[0] == '-s':
        genome_chrom_sizes_path = i[1]
    if i[0] == '-c':
        cells = i[1]
    if i[0] == '-d':
        distance = int(i[1])
    if i[0] == '-f':
        feature_dir = i[1]

# Validate enhancer path
if enhancer_path == "":
    showArgError(parser)
    exit()
elif not os.path.isfile(enhancer_path):
    showError('The following file path for human enhancers does not exist' + '\n' + enhancer_path)
    exit()

# Validate gene path
if not os.path.isfile(gene_path):
    showError('The following file path for human genes does not exist' + '\n' + gene_path)
    exit()

# Validate chromosome sizes file path
if not os.path.isfile(genome_chrom_sizes_path):
    showError('The following file path for chromosome sizes of human genome does not exist' + '\n' + genome_chrom_sizes_path)
    exit()

# Separate new cell lines from default cell lines
default_cell_lines = cells_default.split(",")
cell_lines = cells.split(",")
new_cell_lines = []
for c in cell_lines:
    if c not in default_cell_lines:
        if feature_dir == 'None':
            showError('Feature directory path not provided for cell line ' + c)
            exit()
        new_cell_lines.append(c)

cell_lines = list(set(cell_lines)-set(new_cell_lines))

# Validate new features
new_features = []
if feature_dir != "None":
    if not os.path.isdir(feature_dir):
        showError('The following feature directory path does not exist' + '\n' + feature_dir)
        exit()
    else:
        feature_file_paths = os.listdir(feature_dir)

    # Separate new features for the new cell lines
    new_cell_lines = []

    feature_list = features.split(',')

    for item in feature_file_paths:

        c = item.split('_')[1]
        feature_name = item.split('_')[2]
        feature_ext = item.split('.')[-1]

        new_feature_dir = DATA_FOLDER + '/FeaturesPeakFiles/' + c

        if feature_name not in feature_list:
            showError('The feature name of the feature file ' + item + ' does not belong to the specified feature list' + '\n{' + features + '}')
            exit()

        if feature_ext not in ['bed', 'narrowPeak', 'broadPeak']:
            showError('The following feature file path must be in .bed or .narrowPeak or .broadPeak format' + '\n' + item)
            exit()
        
        if not os.path.isdir(new_feature_dir):
            os.mkdir(new_feature_dir)
                
        copyfile(feature_dir + '/' + item, new_feature_dir + '/' + item)

        new_features.append(feature_name)

        new_cell_lines.append(c)

    cell_lines = list(set(cell_lines).union(set(new_cell_lines)))

# Save enhancers
enhancers = readFileInTable(enhancer_path)
enhancers = [item[:3] + ['_'.join(item[:3]), 1, '.'] for item in enhancers]
writeDataTableAsText(enhancers, DATA_FOLDER + "/" + enhancer_path_system + "/enhancers")

# Save promoters
getPromoters(gene_path, genome_chrom_sizes_path)

# Create ep-pairs
promoters = readFileInTable(DATA_FOLDER + "/" + promoter_path_system + "/promoters_with_genes.bed")
#promoters = promoters[:100]
promoters_gene_dict = dict([["_".join(item[:3]), item[3]] for item in promoters])
ep_pairs = generateEPPairs(enhancers, promoters, distance)

for c in cell_lines:
    writeDataTableAsCSV([["name_E", "name_P", "label"]] + ep_pairs, DATA_FOLDER + "/" + ep_pairs_path_system + "/" + c + "_pairs")

# Create customized config file
config = "Cells:" + ",".join(sorted(cell_lines)) + \
         "\nFeatures:" + features + \
         "\nNumOfCells:" + str(len(cell_lines)) + \
         "\nPairs_path:" + ep_pairs_path_system + \
         "\nEnhancers_path:" + enhancer_path_system + \
         "\nPromoters_path:" + promoter_path_system

writeFile(CONFIG_FP_CUSTOMIZED, config)

# Run feature extraction program
res = os.system(curr_dir + "/ExtractFeatures.sh " + DATA_FOLDER + " " + CONFIG_FP_CUSTOMIZED + " " + ALLOW_WINDOW_FEATURES + " " + curr_dir)

if res != 0:
    exit()

# Predict ep-pairs with features    
model_file_balanced = "model_30_perc_8_cell_lines_30_balanced_hard.pkl"
model_file_unbalanced = "model_30_perc_8_cell_lines_30_unbalanced_hard.pkl"

predict(model_file_balanced, model_file_unbalanced)
