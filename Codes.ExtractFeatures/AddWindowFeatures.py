import Utils
import pandas as pd
import sys
from common import printDec

printDec("AddWindowFeatures Start")

data_dir = str(sys.argv[1])#
config_fp = str(sys.argv[2])
configs = Utils.read_config_file(config_fp)
cells = configs['Cells'].split(',')
EPW_feats_dir =  data_dir + "/EPW_Features"
Utils.make_dir_rename_if_exists(EPW_feats_dir)
temp_feats_dir = data_dir + "/TempFeats"


for cell_name in cells:
    cell_dir = "%s/%s_temp_feats" % (temp_feats_dir, cell_name)  
    
    window_feats_file = "%s/aggregated_mean_%s_W.csv" % (cell_dir, cell_name)
    EP_feats_file = "%s/features_mean_signals_EP_%s.csv" % (cell_dir, cell_name)
    EPW_feats_file = "%s/%s_EPW_feats.csv" % (EPW_feats_dir, cell_name)
    
    windows_df = pd.read_csv(window_feats_file)
    EP_df = pd.read_csv(EP_feats_file)
    
    print(windows_df.shape)
    print(EP_df.shape)
    
    windows_df.reset_index()
    EP_df.reset_index()
    
    print("EP", EP_df.columns)
    
    merged_df = pd.merge(windows_df, EP_df, how='outer', on='name')
    
    print("W added", merged_df.shape)
    
    merged_df = merged_df.fillna(0)
    merged_df.reset_index()
    merged_df.to_csv(EPW_feats_file, index=False)
    
printDec("AddWindowFeatures End")
    
