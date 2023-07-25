from Utils import *
# import pybedtools
import pandas as pd
import Utils
import sys
import pdb

from common import readFileInTable, getOverlappedPeaks2, printDec

printDec("CalculateFeatures Start")

data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])
show_window_features = str(sys.argv[3])

# -----------------------------------------------


def construct_intersection_header(peak_type, reg_type):
    header = []
    e_header = ["chrom_E", "chromStart_E", "chromEnd_E", "name_E", "score_E", "strand_E"]  # "signalValue_E", "pValue_E", "qValue_E", "peak_E"]
    p_header = ["chrom_P", "chromStart_P", "chromEnd_P", "name_P", "score_P", "strand_P"]  # "signalValue_P", "pValue_P", "qValue_P", "peak_P"]
    w_header = ["chrom_W", "chromStart_W", "chromEnd_W", "name_W"]
    narrow_header = ["chrom_F", "chromStart_F", "chromEnd_F", "name_F", "score_F", "strand_F", "signalValue_F", "pValue_F", "qValue_F", "peak_F"]
    broad_header = ["chrom_F", "chromStart_F", "chromEnd_F", "name_F", "score_F", "strand_F", "signalValue_F", "pValue_F", "qValue_F"]
    bed_header = ["chrom_F", "chromStart_F", "chromEnd_F", "name_F", "strand_F"]

    if (reg_type == 'E'):
        if (peak_type == "broadPeak"):
            header = e_header + broad_header
        elif (peak_type == "narrowPeak"):
            header = e_header + narrow_header
        elif (peak_type == "bed"):
            header = e_header + bed_header
    elif (reg_type == 'P'):
        if (peak_type == "broadPeak"):
            header = p_header + broad_header
        elif (peak_type == "narrowPeak"):
            header = p_header + narrow_header
        elif (peak_type == "bed"):
            header = p_header + bed_header
    elif (reg_type == 'W'):
        if (peak_type == "broadPeak"):
            header = w_header + broad_header
        elif (peak_type == "narrowPeak"):
            header = w_header + narrow_header
        elif (peak_type == "bed"):
            header = w_header + bed_header

    return header

# -----------------------------------------------


def peak_file_characteristics(peak_fn):
    peak_type = ""
    if "broadPeak" in peak_fn:
        peak_type = "broadPeak"
    elif "narrowPeak" in peak_fn:
        peak_type = "narrowPeak"
    elif "bed" in peak_fn:
        peak_type = "bed"
    else:
        raise ValueError('The file format is not supported')

    feat_name = ''
    cell_name = ''
    splt = peak_fn.split('_')
    print(splt)
    if len(splt) == 4:
        feat_name = splt[2]
        cell_name = splt[1]

    return peak_type, feat_name, cell_name

# -----------------------------------------------


def intersect_feat_reg(feat_fp, reg_fp, feat_name, peak_type, reg_type):

    data_feat = readFileInTable(feat_fp)
    data_reg = readFileInTable(reg_fp)

    intersection_headers = construct_intersection_header(peak_type, reg_type)
    intersection_df = []
    overlapped_indices = getOverlappedPeaks2(data_reg, data_feat)
    for i in sorted(overlapped_indices.keys()):
        reg = data_reg[i]
        for j in overlapped_indices[i]:
            intersection_df.append(reg + data_feat[j])

    return pd.DataFrame(intersection_df, columns=intersection_headers)

    # pybedtools removed
    '''
    reg_bt = pybedtools.BedTool(reg_fp)
    feat_bt = pybedtools.BedTool(feat_fp)
    try:
        intersection = reg_bt.intersect(feat_bt, wa=True, wb=True)    
        intersection_header = construct_intersection_header(peak_type, reg_type)
            
        intersection_df = intersection.to_dataframe()
        intersection_df.columns = intersection_header
    except:
	    pdb.set_trace()    
    return intersection_df
    '''

# -----------------------------------------------


def save_feature(intersection_df, reg_type, feats_dir, feat_name, cell_name):
    # reg_lb = 'E' if is_enhancer == True else 'P'
    f_name = 'name_' + reg_type
    feat_fp = "%s/%s_%s_%s.csv" % (feats_dir, cell_name, feat_name, reg_type)

    intersection_df['signalValue_F'] = intersection_df['signalValue_F'].astype('float64')
    grouped = intersection_df.loc[:, [f_name, 'signalValue_F']].groupby(f_name).mean()

    indexed_grp = grouped.add_suffix('_mean').reset_index()
    signals_df = indexed_grp[[f_name, "signalValue_F_mean"]]
    featColName = "%s_%s" % (feat_name, reg_type)
    signals_df.columns = [f_name, featColName]
    signals_df.to_csv(feat_fp, index=False)
    print(feat_fp)

# -----------------------------------------------


def main():
    configs = Utils.read_config_file(config_fp)
    cell_names = configs['Cells'].split(',')
    features = configs['Features'].split(',')
    enhancers_dir = data_dir + "/" + configs['Enhancers_path']
    promoters_dir = data_dir + "/" + configs['Promoters_path']
    peaks_dir = data_dir + "/FeaturesPeakFiles"
    windows_dir = data_dir + "/Windows"
    temp_feats_dir = data_dir + "/TempFeats"
    Utils.make_dir_rename_if_exists(temp_feats_dir)

    print(cell_names)

    for cell_name in cell_names:
        peaks_path = "%s/%s" % (peaks_dir, cell_name)
        enhancer_fp = "%s/enhancers" % (enhancers_dir)
        promoter_fp = "%s/promoters" % (promoters_dir)
        window_fp = "%s/%s_windows.intervals" % (windows_dir, cell_name)
        feats_dir = "%s/%s_temp_feats" % (temp_feats_dir, cell_name)

        enhancer_feats_dir = "%s/feats_E" % (feats_dir)
        promoter_feats_dir = "%s/feats_P" % (feats_dir)
        window_feats_dir = "%s/feats_W" % (feats_dir)
        Utils.make_dir_rename_if_exists(feats_dir)
        Utils.make_dir_rename_if_exists(enhancer_feats_dir)
        Utils.make_dir_rename_if_exists(promoter_feats_dir)
        Utils.make_dir_rename_if_exists(window_feats_dir)
        files = os.listdir(peaks_path)

        for fn in files:
            if not os.path.isfile(os.path.join(peaks_path, fn)):
                continue

            peak_type, feat_name, cell_name = peak_file_characteristics(fn)

            if feat_name not in features:
                continue

            feat_fp = "%s/%s" % (peaks_path, fn)

            # Enhancer
            intersection_df = intersect_feat_reg(feat_fp, enhancer_fp, feat_name, peak_type, reg_type='E')
            save_feature(intersection_df, 'E', enhancer_feats_dir, feat_name, cell_name)

            # Promoter
            intersection_df = intersect_feat_reg(feat_fp, promoter_fp, feat_name, peak_type, reg_type='P')
            save_feature(intersection_df, 'P', promoter_feats_dir, feat_name, cell_name)

            if show_window_features == "1":
                # Window
                intersection_df = intersect_feat_reg(feat_fp, window_fp, feat_name, peak_type, reg_type='W')
                save_feature(intersection_df, 'W', window_feats_dir, feat_name, cell_name)

    printDec("CalculateFeatures End")


main()
