# EPIP

Author: Amlan Talukder

Date: May 28,2018

EPIP is a software used to identify target genes of enhancers in human genome. It is developed by the computational System biology group at UCF.


INSTALLATION
--------------------------------------------------------------------------------------------
   1. Install python 3.7
   2. Run the following commands to install python packages

		```
		pip install -r requirements.txt
		```
	

EXECUTION 
--------------------------------------------------------------------------------------------------------------------------------------

   1. cd (change directory) to the "EPIP/Codes.ExtractFeatures" directory of the software. 
   2. Make sure the "python" environment variable points to the path of python version 3.5.
   3. You can run the software by running the script "EPIP.py" with the following command:
   
   ----------------------------------------------------------------------------------------
   
	usage: python EPIP.py [-h] -e ENHANCER_FILE_PATH [-c CELL_LINE] [-g GENE_FILE_PATH]
               [-s CHROM_SIZES_FILE_PATH] [-d INTEGER] [-f FEATURE_DIR_PATH]

	Find enhancer targets

	optional arguments:
	  -h, --help            show this help message and exit
	  -c CELL_LINE          Comma delimited cell lines among
		                {GM12878,HELA,HMEC,HUVEC,IMR90,K562,KBM7,NHEK}.In
		                order to predict for a cell line that is not in the
		                list above, user must provide the file path of the
		                features to use for that cell line with "-f"
		                command
	  -g GENE_FILE_PATH     Path for human genes in bed file format (default: Use
		                human Gencode 19)
	  -s CHROM_SIZES_FILE_PATH
		                Path for chromosome sizes file of human genome in tab
		                delimited file format (default: Use human Gencode 19
		                chrom sizes)
	  -d INTEGER            Maximum distance between enhancers and promoters
		                (default: 2 Mbp)
	  -f FEATURE_DIR_PATH   Path of the directory where the feature peak files for
		                each cell line are specified in the following formats 
		                feature_<cellline>_<feature_name>_peaks.[bed|narrowPea
		                k|broadPeak]. <feature_name> must be among the
		                following features. {H3k4me1,H3k4me2,H3k4me3,H3k9ac,H3
		                k27ac,H3k27me3,H3k36me3,H3k79me2,H4k20me1,Ctcf,DNaseI,
		                Pol2,Rad21,Smc3}

	required arguments:
	  -e ENHANCER_FILE_PATH
		                Path for hg19 enhancers in bed file format

	Example: python EPIP.py -e ../test_enhancers.bed -g ../test_genes.bed

Required inputs
---------------------------------------------------------------------------------------------
The tool takes the file path of the enhancer bed file as mandatory parameters.

Optional inputs
---------------------------------------------------------------------------------------------
The cell lines, gene file paths, chromosome sizes file path, distance and feature dir paths are the optional input parameters. 
In case of multiple cell lines as input, the cell lines input has to be comma delimited. Any cell line except the listed cell lines, 
can be taken as input. For the new cell line, feature files should be stored in a directory specified by the user using the FEATURE_DIR_PATH 
parameter. A example feature directory "Example_feature_dir" is given in the current folder. If no features specified by the user for the 
new cell line, only the basic features (distance, corr and css) will be calculated for that cell line. The features specified for the new 
cell line must be among the list of following features {H3k4me1,H3k4me2,H3k4me3,H3k9ac,H3k27ac,H3k27me3,H3k36me3,H3k79me2,H4k20me1,Ctcf,
DNaseI,Pol2,Rad21,Smc3}. Although user can specify different data file for a feature among the listed features, using the feature reference 
file parameter. If the genes bed file is provided as an input, the promoters are calculated using 1kbp upstream and 100bp downstream around 
the tss of the genes. The enhancer promoter pairs are considered within a specific distance (default: 2Mbp). This distance can also be taken 
as an input.


MODEL
----------------------------------------------------------------------------------------------------------------------------------
The model pkl file is stored under "TrainedModels" directory.


RESULTS
----------------------------------------------------------------------------------------------------------------------------------
The result files are stored under "Results" directory. The result files are created by the name of the cell lines.
The output format is tab delimited enhancer regions and target gene ids predicted by the model. The example of the output format is following,

	chr1_2989176_2989511    ENSG00000169717.5_ACTRT2
	chr1_2230678_2230895    ENSG00000116151.9_MORN1
	chr1_2059932_2059994    ENSG00000234396.3_RP11-181G12.4
	chr1_3581269_3581645    ENSG00000272088.1_RP11-168F9.2
	chr1_1978731_1978937    ENSG00000169885.5_CALML6
	chr1_2231691_2232185    ENSG00000178642.5_AL513477.1
	chr1_1005293_1005547    ENSG00000207607.1_MIR200A
	chr1_1136075_1136463    ENSG00000260179.1_RP5-902P8.12
	chr1_2246556_2246763    ENSG00000243558.1_RP11-181G12.5


LICENSE & CREDITS
-------------------------------------------------------------------------------------------------
The software is a freely available for academic use.
plase contact xiaoman shawn li (xiaoman@mail.ucf.edu) for further information. 


CONTACT INFO
-------------------------------------------------------------------------------------------------
If you are encountering any problem regarding to EPIP, please refer the manual first.
If problem still can not be solved, please feel free to contact us:
Amlan Talukder (amlan@knights.ucf.edu)
Nancy Haiyan Hu (haihu@cs.ucf.edu)
Xiaoman Shawn Li (xiaoman@mail.ucf.edu)
