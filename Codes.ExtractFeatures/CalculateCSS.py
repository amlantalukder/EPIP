import sys,os,getopt
from common import *
import Utils

# -----------------------------------------------
def alignEnhancers(enhancers_file, enhancers_alignment_path, aligned_enhancers_path):

    print("Aligning enhancers with 5 other species...")

    os.system(curr_dir + '/LiftOver_3/liftOver -minMatch=0.1 ' + enhancers_file + ' ' + enhancers_alignment_path + '/hg19ToGalGal3.over.chain.gz ' + aligned_enhancers_path + '/chicken_enhancers_align_galGal3.bed ' + aligned_enhancers_path + '/unlifted_chicken.bed')
    os.system(curr_dir + '/LiftOver_3/liftOver -minMatch=0.1 ' + enhancers_file + ' ' + enhancers_alignment_path + '/hg19ToDanRer7.over.chain.gz ' + aligned_enhancers_path + '/zebrafish_enhancers_align_zv9.bed ' + aligned_enhancers_path + '/unlifted_zeb.bed')
    os.system(curr_dir + '/LiftOver_3/liftOver -minMatch=0.1 ' + enhancers_file + ' ' + enhancers_alignment_path + '/hg19ToMm10.over.chain.gz ' + aligned_enhancers_path + '/mouse_enhancers_align_mm10.bed ' + aligned_enhancers_path + '/unlifted_mm.bed')
    os.system(curr_dir + '/LiftOver_3/liftOver -minMatch=0.1 ' + enhancers_file + ' ' + enhancers_alignment_path + '/hg19ToPanTro4.over.chain.gz ' + aligned_enhancers_path + '/chimp_enhancers_align_panTro4.bed ' + aligned_enhancers_path + '/unlifted_chimp.bed')
    os.system(curr_dir + '/LiftOver_3/liftOver -minMatch=0.1 ' + enhancers_file + ' ' + enhancers_alignment_path + '/hg19ToXenTro3.over.chain.gz ' + aligned_enhancers_path + '/frog_enhancers_align_xenTro3.bed ' + aligned_enhancers_path + '/unlifted_frog.bed')

# -----------------------------------------------
#load genelocation Data
def loadgeneLocations(species_path):
    
    chimp_genes = readFileInTable(species_path + "/chimp_gene_Pan_troglodytes-2.1.4.bed")
    chimp_genes_dict = dict([[item[3].upper(), item] for item in chimp_genes])
    
    mouse_genes = readFileInTable(species_path + "/mouse_gene_mm10.bed")
    mouse_genes_dict = dict([[item[3].upper(), item] for item in mouse_genes])
    
    zebfish_genes = readFileInTable(species_path + "/zebrafish_gene_zv9.bed")
    zebfish_genes_dict = dict([[item[3].upper(), item] for item in zebfish_genes])
    
    chicken_genes = readFileInTable(species_path + "/chicken_gene_Gallus_gallus-4.0.bed")
    chicken_genes_dict = dict([[item[3].upper(), item] for item in chicken_genes])

    frog_genes = readFileInTable(species_path + "/frog_gene_xenTro3.bed")
    frog_genes_dict = dict([[item[3].upper(), item] for item in frog_genes])

    print("# of chimpanzee genes:", len(chimp_genes))
    print("# of mouse genes:", len(mouse_genes))
    print("# of zebrafish genes:", len(zebfish_genes))
    print("# of chicken genes:", len(chicken_genes))
    print("# of frog genes:", len(frog_genes))
        
    return [chimp_genes_dict, mouse_genes_dict, zebfish_genes_dict, chicken_genes_dict, frog_genes_dict]

# -----------------------------------------------
def loadAlignedEnhancers(aligned_enhancers_path):

    chimp_enhancers = readFileInTable(aligned_enhancers_path + "/chimp_enhancers_align_panTro4.bed")
    chimp_enhancers_dict = dict([[item[3], item] for item in chimp_enhancers])
    
    mouse_enhancers = readFileInTable(aligned_enhancers_path + "/mouse_enhancers_align_mm10.bed")
    mouse_enhancers_dict = dict([[item[3], item] for item in mouse_enhancers])
    
    zebfish_enhancers = readFileInTable(aligned_enhancers_path + "/zebrafish_enhancers_align_zv9.bed")
    zebfish_enhancers_dict = dict([[item[3], item] for item in zebfish_enhancers])
    
    chicken_enhancers = readFileInTable(aligned_enhancers_path + "/chicken_enhancers_align_galGal3.bed")
    chicken_enhancers_dict = dict([[item[3], item] for item in chicken_enhancers])
    
    frog_enhancers = readFileInTable(aligned_enhancers_path + "/frog_enhancers_align_xenTro3.bed")
    frog_enhancers_dict = dict([[item[3], item] for item in frog_enhancers])
        
    return [chimp_enhancers_dict, mouse_enhancers_dict, zebfish_enhancers_dict, chicken_enhancers_dict, frog_enhancers_dict]

def getDistance(e, p):

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

#----------------------------------------------------------------------

data_dir = str(sys.argv[1])
config_fp = str(sys.argv[2])
curr_dir = str(sys.argv[3])

#data_dir = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/EPIP/Data"
#config_fp = "/home/amlan/Lab_projects/Project6/Samaneh/EPIP/EPIP/Configs/config_temp"

configs = Utils.read_config_file(config_fp)
cell_lines = configs['Cells'].split(',')
pairs_path = data_dir + "/" + configs['Pairs_path']
enhancers_path = data_dir + "/" + configs['Enhancers_path']
promoters_path = data_dir + "/" + configs['Promoters_path']

human_enhancers = readFileInTable(enhancers_path + "/enhancers")
alignEnhancers(enhancers_path + "/enhancers", enhancers_path + "/enhancers_alignment", enhancers_path + "/Six_species")
aligned_enhancers = loadAlignedEnhancers(enhancers_path + "/Six_species")

# species
species = ["chimpanzee", "mouse", "zebrafish", "chicken", "frog"]

# Phylogenetic distances between humand and each one of the species
phy_distance = [0.02, 0.46, 1.83, 1.1, 1.43]

# Make a data structure where each promoter points to gene index
print("Mapping promoter to gene file index ...")
human_genes = readFileInTable(promoters_path + "/Six_species/human_gene_hg19.bed")
human_promoters = readFileInTable(promoters_path + "/promoters")

human_genes.sort(cmp=comparePeaks)
human_promoters_dict = {}

overlapped_indices = getOverlappedPeaks2(human_promoters, human_genes)

for i, p in enumerate(human_promoters):
    if i in overlapped_indices:
        human_promoters_dict["_".join(p[:3])] = overlapped_indices[i]

# Load other species genes
ext_gene_loc = loadgeneLocations(promoters_path + "/Six_species/")

for c in cell_lines:

    css_output = []

    human_EP_pairs = readFileInTable(pairs_path + "/" + c + "_pairs", delim=",")
    
    output_header = human_EP_pairs[0] + ['CSS']

    for ep in human_EP_pairs[1:]:

        e_key = ep[0]
        p_key = ep[1]

        # Get gene info from promoter key
        gene_indices = human_promoters_dict[p_key]

        score = 0

        for i in range(len(species)):

            if e_key in aligned_enhancers[i]:
    
                aligned_enhancer = aligned_enhancers[i][e_key]

                for gene_index in gene_indices:

                    g = human_genes[gene_index]
                    eg_distance = getDistance(e_key.split("_"), g)
                    
                    gene_id = g[3]

                    if gene_id.upper() in ext_gene_loc[i]:
                        ext_gene_id = ext_gene_loc[i][gene_id.upper()]
                
                        distance = getDistance(aligned_enhancer, ext_gene_id)

                        #if e_key + '-' + p_key == 'chr5_79543086_79543324-chr5_79535308_79536408':
                         #   print distance, gene_id, species[i], ext_gene_id

                        #if distance <= 2000000: #eg_distance:
                        score += phy_distance[i]
                        break

        #if e_key + '-' + p_key == 'chr5_79543086_79543324-chr5_79535308_79536408':
         #   print c, score
            
        css_output.append(ep + [score])

    writeDataTableAsCSV([output_header] + css_output, data_dir + "/CSS/" + c + "_pairs_css")
        
                    

