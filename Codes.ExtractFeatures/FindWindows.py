import pandas as pd
import sys
import Utils

print "FindWindows", "Start"

data_dir = str(sys.argv[1])
#data_dir = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Data"
config_fp = str(sys.argv[2])
#config_fp = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/Configs/config_extract_features"

def main():
    configs = Utils.read_config_file(config_fp)
    cutoff = configs['Cutoff']
    pairs_dir = data_dir + "/EP-Pairs/test"
    windows_dir = data_dir + "/Windows"
    cell_names = configs['Cells'].split(',')
    Utils.make_dir_rename_if_exists(windows_dir)           
    
    for cn in cell_names:
        print cn
        pairs_file = "%s/%s_pairs_%s" % (pairs_dir, cn, cutoff)
        windows_file = "%s/%s_windows.intervals" % (windows_dir, cn)
        
        pairs_df = pd.read_csv(pairs_file)
        windows_df = pd.DataFrame(columns=["chrom", "chromStart", "chromEnd", "name"])
	perc = 10
	counter = 0
        for i, r in pairs_df.iterrows():
            e_name = r['name_E']
            p_name = r['name_P']
            window_info = find_window(e_name, p_name)
	    if window_info == ['chr0', 0, 1]:
		continue
            window_name = "%s-%s" % (e_name, p_name)
            window_row = window_info + [window_name]
            windows_df.loc[counter] = window_row
	    counter += 1

	    if (i+1)*100/len(pairs_df) >= perc:
		print(perc, "%")	    	
		perc += 10

        Utils.write_bed_file(windows_file, windows_df) 
    print "End"
            
def find_window(e_name, p_name):
    [e_chr, e_start, e_end] = e_name.split('_')
    [p_chr, p_start, p_end] = p_name.split('_') 
    if (e_chr != p_chr):
        return ['chr0', 0, 1]
    w_chr = e_chr
    w_start = min([int(e_end), int(p_end)])
    w_end = max([int(e_start), int(p_start)])
    if (w_end > w_start):
        return [w_chr, str(w_start), str(w_end)]
    return ['chr0', 0, 1]
  
    

main()
