import pandas as pd
import os
import sys
import Utils, pdb

print "AggregateFeatures", "Start"


data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])
show_window_features = str(sys.argv[3])
configs = Utils.read_config_file(config_fp)
cells = configs['Cells'].split(',')
temp_feats_dir = data_dir + "/TempFeats"

if show_window_features == "1":
	regs = ['E', 'P', 'W']
else:
	regs = ['E', 'P']

for cell_name in cells:
    for reg_lbl in regs:                      
        name_f = 'name_' + reg_lbl
        feats_dir = "%s/%s_temp_feats/feats_%s" % (temp_feats_dir, cell_name, reg_lbl)
        feature_files = os.listdir(feats_dir)
        outFile = "%s/%s_temp_feats/aggregated_mean_%s_%s.csv" % (temp_feats_dir, cell_name, cell_name, reg_lbl)

        feats_dfs = []
        for fn in feature_files:
            fp = feats_dir + "/" + fn
            feats_dfs.append(pd.read_csv(fp, index_col=False))

        if len(feats_dfs) == 0:
	    continue

	merged_df = feats_dfs[0]
	for i in range(1, len(feats_dfs)):
            merged_df = pd.merge(merged_df, feats_dfs[i], how='outer', on=name_f)
        
        merged_df = merged_df.fillna(0)
        
        if (name_f == 'name_W'):
            merged_df.rename(columns={name_f:'name'}, inplace=True)
            merged_df.to_csv(outFile, index=False)
        else:
            merged_df.to_csv(outFile, index=False)
print "End"
