import pandas as pd
import os
import shutil
import datetime

# -----------------------------------------------
def read_riple_file(fn):
    return pd.read_csv(fn, sep = r'\s+', index_col = False)

# -----------------------------------------------
def read_bed_file(fn):
    return pd.read_csv(fn, sep = r'\t', index_col = False)

# -----------------------------------------------
def write_bed_file(file_path, dataframe):
    dataframe.to_csv(file_path, sep='\t', index=False, header=False)

# -----------------------------------------------
def read_hic_file(fn):
    return pd.read_csv(fn, sep = r'\t', index_col = False)

# -----------------------------------------------
def getsubgrid(x1, y1, x2, y2, grid):
    return [item[x1:x2] for item in grid[y1:y2]]

# -----------------------------------------------
def make_dir_rename_if_exists(dir_path):    
    if (os.path.isdir(dir_path)):
        new_dir = "%s_%s" % (dir_path, datetime.datetime.now().strftime("%Y-%B-%d_%I-%M%p"))
        os.renames(dir_path, new_dir)
        #shutil.rmtree(dir_path)
    os.mkdir(dir_path)

# -----------------------------------------------
def read_config_file(config_fp):
    cfin = open(config_fp, 'r')
    clines = cfin.readlines()
    configs = dict([l.strip().split(':') for l in clines])   
    return configs