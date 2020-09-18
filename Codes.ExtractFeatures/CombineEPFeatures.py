import pandas as pd
from time import sleep
import sys, os, pdb
import Utils

print "CombineEPFeatures", "Start"

#data_dir = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Data"
#config_fp = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Configs/config_extract_features"
data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])
configs = Utils.read_config_file(config_fp)
temp_feats_dir = data_dir + "/TempFeats"
css_dir = data_dir + "/CSS"
pairs_dir = data_dir + "/" + configs['Pairs_path']
cells = configs['Cells'].split(',')

def extract_column_names(feat_cols):
    cols = ['name'] + feat_cols + ['corr', 'distance', 'css', 'label']
    #cols = ['name'] + feat_cols + ['len_E', 'len_P', 'len_W', 'label']
    return cols

def calculateSpearmanCorrelation(e_dhs, p_dhs):

    n = len(e_dhs)

    e_dhs_rank = sorted(range(n), key=lambda x: e_dhs[x])
    p_dhs_rank = sorted(range(n), key=lambda x: p_dhs[x])

    sum_d_sqr = sum([(e_dhs_rank[i]-p_dhs_rank[i]) * (e_dhs_rank[i]-p_dhs_rank[i]) for i in range(n)])

    return 1-(6.0 * sum_d_sqr)/(n*(n*n-1))

def getDistance(e, p):

    e = e.split("_")
    p = p.split("_")

    if e[0] == p[0]:

        # Enhancer is downstream of the gene
        if int(e[1]) > int(p[2]):
            return int(e[1]) - int(p[2])

        # Enhancer is upstream of the gene
        if int(p[1]) > int(e[2]):
            return int(p[1]) - int(e[2])

        # Enhancer is ovelapped with the gene
        return 0

    return -1

def getEPLength(e, p):

    len_W = getDistance(e, p)
    e = e.split("_")
    len_E = int(e[2])-int(e[1])
    p = p.split("_")
    len_P = int(p[2])-int(p[1])

    return len_E, len_P, len_W

dhs_enhancers_df = pd.read_csv(data_dir + "/Enhancer_Features/enhancer_dhs")
dhs_promoters_df = pd.read_csv(data_dir + "/Promoter_Features/promoter_dhs")
dhs_enhancers_dict = dhs_enhancers_df.set_index('name_E').T.to_dict('list')
dhs_promoters_dict = dhs_promoters_df.set_index('name_P').T.to_dict('list')

extra_feat_dict = {}

for cell_name in cells:
    cell_dir = "%s/%s_temp_feats" % (temp_feats_dir, cell_name)
    E_features_file = "%s/aggregated_mean_%s_E.csv" % (cell_dir, cell_name)
    P_features_file = "%s/aggregated_mean_%s_P.csv" % (cell_dir, cell_name)
    pairs_file = "%s/%s_pairs" % (pairs_dir, cell_name)
    combined_EPfeats_file = "%s/features_mean_signals_EP_%s.csv" % (cell_dir, cell_name)
    css_features_file = "%s/%s_pairs_css" % (css_dir, cell_name)

    pairs_df = pd.read_csv(pairs_file)
    
    if (os.path.isfile(E_features_file)):
        E_feat_df = pd.read_csv(E_features_file)
	E_feat_dict = E_feat_df.set_index('name_E').T.to_dict('list')
        E_cols = list(E_feat_df.columns)[1:]
    else:
        E_feat_dict = {}
	E_cols = []        

    if (os.path.isfile(P_features_file)):
        P_feat_df = pd.read_csv(P_features_file)   
        P_feat_dict = P_feat_df.set_index('name_P').T.to_dict('list')
        P_cols = list(P_feat_df.columns)[1:]
    else:
        P_feat_dict = {}
        P_cols = []

    css_feat_df = pd.read_csv(css_features_file)
    css_feat_dict = dict(zip(zip(css_feat_df.name_E, css_feat_df.name_P), css_feat_df.CSS))
    
    feats_columns = extract_column_names(E_cols + P_cols)

    print cell_name
    print feats_columns

    default_column_values = [0] * len(E_cols)
    
    interactions_df = pd.DataFrame(columns=feats_columns)
    interactions_df.to_csv(combined_EPfeats_file, index=False, mode="a+")

    perc = 10

    for i, r in pairs_df.iterrows():
        e = r['name_E']
        p = r['name_P']

        if e in E_feat_dict:
            E_feat = E_feat_dict[e] 
        else:
            E_feat = default_column_values
            
        if p in P_feat_dict:
            P_feat = P_feat_dict[p]
        else:
            P_feat = default_column_values

	if e + '-' + p not in extra_feat_dict:
        	dhs_corr = calculateSpearmanCorrelation(dhs_enhancers_dict[e], dhs_promoters_dict[p])
		distance = getDistance(e, p)
		extra_feat_dict[e + '-' + p] = [dhs_corr, distance]
	else:
		dhs_corr = extra_feat_dict[e + '-' + p][0]
		distance = extra_feat_dict[e + '-' + p][1]
        
        css = css_feat_dict[(e,p)]
        
	#len_E, len_P, len_W = getEPLength(e, p)
 
        interaction_label = r['label']
        interaction_feat = E_feat + P_feat + [dhs_corr, distance, css]  # I assumed this order in writing extract_column_names function.
                                            # If this order is changed extract_column_names should be changed.
	#interaction_feat = E_feat + P_feat + [len_E, len_P, len_W]
        interaction_name = "%s-%s" % (e, p)
        interaction_row = [interaction_name] + interaction_feat + [interaction_label]
	try:
        	interactions_df.loc[i] = interaction_row
	except:
		pdb.set_trace()

        if (i+1)*100/len(pairs_df.index) >= perc:
	    interactions_df.to_csv(combined_EPfeats_file, index=False, header=False, mode="a+")
	    interactions_df = pd.DataFrame(columns=feats_columns)
            print perc, '%'
            perc += 10
        
    
print "End"
