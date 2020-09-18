#import xlsxwriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from itertools import islice
import os, pdb
from scipy.stats import mannwhitneyu as mwu
import datetime

FILE_READ_MAX_SIZE = 1000000000

#-----------------------------------------------
def readFile(filename):

    fl = open(filename, "r")

    data = fl.readlines()

    fl.close()

    return data

#-----------------------------------------------
def readFileInTable(filename, delim='\t'):

    fl = open(filename, "r")

    data = fl.readlines()

    fl.close()

    data = [item.strip().split(delim) for item in data]

    return data


#-----------------------------------------------
def readBigFile(filename):

    fl = open(filename, "r")

    returnData = []

    data = fl.readlines(FILE_READ_MAX_SIZE)

    while(data):

        returnData += [item.replace("\n", "").split("\t") for item in data]

        data = fl.readlines(FILE_READ_MAX_SIZE)
            
    fl.close()

    return returnData

#-----------------------------------------------
def readBigFileInPart(filename):

    data = []

    with open(file_path) as f:
	while True:
	    file_chunk = list(islice(f, 1000000))
	    if not file_chunk:
		break
	    data += file_chunk
        
#-----------------------------------------------
def readInputFile(filename):

    datafileInfo = os.stat(filename)

    if datafileInfo.st_size > FILE_READ_MAX_SIZE:

        data = readBigFile(filename)

    else:

        data = readFile(filename)

    return data

#-----------------------------------------------
def readPartFile(filename, sizeInBytes):

    fl = open(filename, "r")

    data = fl.readlines(sizeInBytes)
    
    fl.close()

    return data

#-----------------------------------------------
def writeFile(filename, data, mode="w"):

    d = os.path.dirname(filename)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

    fl = open(filename, mode)
    fl.write(data)
    fl.close()

#-----------------------------------------------
def createHTMLTable(data, col_size):

    if len(col_size) > 0:
	return "<table border='1'>" + "".join(["<tr>" + "".join(["<td width='" + str(col_size[i]) + "'>" + str(item[i]) + "</td>" for i in range(len(item))]) + "</tr>" for item in arr]) + "</table>"
    else:
	return "<table border='1'>" + "".join(["<tr>" + "".join(["<td>" + str(item1) + "</td>" for item1 in item]) + "</tr>" for item in arr]) + "</table>"

#-----------------------------------------------
def writeHTMLToFile(fileName, arr, col_size):

    writeFile(fileName, createHTMLTable(arr, col_size))
    
#-----------------------------------------------
def writeDataAsHTMLTable(data, filename):

    writeFile(filename + ".html", createHTMLTable(data))

#-----------------------------------------------
def writeDataTableAsText(data, filename, mode="w"):

    text = formatDataTable(data, "\t", "\n")
        
    writeFile(filename, text, mode)

#-----------------------------------------------
def writeDataTableAsCSV(data, filename, mode="w"):

    text = formatDataTable(data, ",", "\n")
        
    writeFile(filename, text, mode)

#-----------------------------------------------
def formatDataTable(data, col_sep="\t", row_sep="\n"):

    return row_sep.join([col_sep.join([str(item1) for item1 in item]) for item in data])

#-----------------------------------------------
def writeData(data, filename, mode="w"):
        
    writeFile(filename, "\n".join([str(item) for item in data]), mode)

#-----------------------------------------------
def writeNewLine(filename, mode="w"):
        
    writeFile(filename, "\n", mode)

#-----------------------------------------------
def getDictDataAsText(data, delim):

    text = ""

    for key in data:

        if isinstance(data[key], list):

            valueText = getArrayDataAsText(data[key], delim)

        elif isinstance(data[key], dict):

            valueText = getDictDataAsText(data[key], delim)

        else:

            valueText = "\"" + str(data[key]) + "\""

        text += ("\"" + key + "\":" + valueText + delim) 

    return ("{" + text[0:len(text)-2] + "}")


#-----------------------------------------------
def getArrayDataAsText(data, delim):

    text = ""
            
    for item in data:

        if isinstance(item, list):

            valueText = getArrayDataAsText(item, delim)

        elif isinstance(item, dict):

            valueText = getDictDataAsText(item, delim)

        else:

            valueText = "\"" + str(item) + "\""

        text += (valueText + delim)

    return ("[" + text[0:len(text)-2] + "]")

#-----------------------------------------------
def writeDictAsText(data, filename):

    text = ""

    for key in data:

        if isinstance(data[key], list):

            valueText = getArrayDataAsText(data[key], ", ")

        elif isinstance(data[key], dict):

            valueText = getDictDataAsText(data[key], ", ")

        else:

            valueText = "\"" + str(data[key]) + "\""

        text += ("\"" + key + "\":" + valueText + ", ") 

    text = "{" + text[0:len(text)-2] + "}"
        
    writeFile(filename, text)

#-----------------------------------------------
def writeArrayAsText(data, filename):

    text = ""

    for item in data:

        if isinstance(item, list):

            valueText = getArrayDataAsText(item, ", ")

        elif isinstance(item, dict):

            valueText = getDictDataAsText(item, delim)

        else:

            valueText = "\"" + str(item) + "\""

        text += (valueText + ", ") 

    text = "[" + text[0:len(text)-2] + "]"
        
    writeFile(filename, text)

#-----------------------------------------------
def find(arr, value, exactMatch = True):
	
	for i in range(len(arr)):
		if (exactMatch and arr[i] == value) or \
		(not exactMatch and arr[i].find(value) >= 0):
			return i
	return -1

#-----------------------------------------------
def find2D(arr, value, col, exactMatch = True):
	
	for i in range(len(arr)):
		if (exactMatch and arr[i][col] == value) or \
		(not exactMatch and arr[i][col].find(value) >= 0):
			return i
	return -1

#-----------------------------------------------
def found(values, valueToMatch):

    for item in values:

        if item == valueToMatch:

            return True

    return False

#-----------------------------------------------
def overlapped(start1, end1, start2, end2):

    start1 = int(start1)
    end1 = int(end1)
    start2 = int(start2)
    end2 = int(end2)

    if (start1 <= start2 and end1 >= start2) or \
       (start2 <= start1 and end2 >= start1):

        return True

    return False

#-----------------------------------------------
def writeExcel(fileName, sheets, sheetData):

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(fileName + '.xlsx')

    for i in range(len(sheets)):

        worksheet = workbook.add_worksheet(sheets[i])
        
        # Some data we want to write to the worksheet.
        data = sheetData[i]

        # Iterate over the data and write it out row by row.
        for row in range(len(data)):

            for col in range(len(data[row])):

                worksheet.write(row, col, data[row][col])

    workbook.close()

#-----------------------------------------------
class CorrGraph:

    #-----------------------------------------------
    def __init__(self, neighbours=None):

        if neighbours != None:
            self.neighbours = neighbours
        else:
            self.neighbours = {}
        self.cliques = []
        self.components = []

    #-----------------------------------------------
    def getConnectedComponents(self):

        num_components = 0
        self.components = []

        visited = dict([[item,0] for item in self.neighbours.keys()])
        
        for v in self.neighbours.keys():
            if visited[v] == 0:
                num_components += 1
                visited[v] = 1
                q = [v]
                new_component = []
                while len(q) != 0:
                    w = q[0]
                    new_component.append(w)
                    q = q[1:]
                    for k in self.neighbours[w]:
                        if visited[k] == 0:
                            visited[k] = 1
                            q.append(k)
                            
                self.components.append(CorrGraph(dict([[item, self.neighbours[item]] for item in new_component if item in self.neighbours])))

        return num_components

    #-----------------------------------------------
    def addEdge(self, n1, n2):

        if n1 not in self.neighbours:
            self.neighbours[n1] = set([n2])
        else:
            self.neighbours[n1].add(n2)
        if n2 not in self.neighbours:
            self.neighbours[n2] = set([n1])
        else:
            self.neighbours[n2].add(n1)

    #-----------------------------------------------
    def getCliques(self):

        self.getConnectedComponents()
        count = 0
        perc = 10
        print "Number of connected components:", len(self.components)
        for g in self.components:
            print "Finding cluster for component with number of nodes ", len(g.neighbours.keys())
            
            if len(g.neighbours.keys()) > 20:
                g.findMaxCliques(set(g.neighbours.keys()), set(), set(), 1, start_time = datetime.datetime.now()) # threshold=len(g.neighbours.keys())/2)
            else:
                g.findMaxCliques(set(g.neighbours.keys()), set(), set(), 1)
                
            self.cliques += sorted(g.cliques, key=lambda x: len(x), reverse=True)
            count += 1
            perc = showPerc(count, len(self.components), perc)
        #self.findMaxCliques(set(self.neighbours.keys()), set(), set(), 1)
        #self.cliques.sort(key=lambda x: len(x), reverse=True)

        return self.cliques

    #-----------------------------------------------
    def findMaxCliques(self, P, R, X, level, verbosearchFilese=False, start_time=None):

        if verbose:
            print "|" + "---"*level, P, R, X

        if len(P.union(X)) == 0:
            if len(R) > 1:
                self.cliques.append(list(R))
                if verbose:
                    print "clique found..."
                #print len(self.cliques)
            return

        if level == 1:
            count = 0
            len_P = len(P)
            perc = 10

        """
        if level > 20:
            return
        """

        if start_time != None and (datetime.datetime.now() - start_time).total_seconds() > 60:
            return
        
        
        for v in P:

            self.findMaxCliques(P.intersection(self.neighbours[v]), R.union(set([v])), X.intersection(self.neighbours[v]), (level + 1), verbose, start_time)

            P = P - set([v])
            X = X.union(set([v]))

            """
            covered_nodes = set()
            for clique in self.cliques:
                covered_nodes = covered_nodes.union(clique)

            if threshold > 0 and len(self.cliques) > 0 and len(covered_nodes) >= threshold:
                return
            """

            #if level == 1:

                #count += 1
                
                #perc = showPerc(count, len_P, perc)


#-------------------------------------------
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#-------------------------------------------
def plotHistogram(data, plot_title, x_label, y_label, x_tick_labels, y_tick_labels):

    fig, ax = plt.subplots(figsize=(14, 12))

    plt.set_cmap('Reds')

    #p = ax.pcolormesh(data)

    ax.pcolor(data)

    ax.set_ylim(0,len(data))                                                                                                               
    ax.set_xlim(0,len(data[0]))
     
    #fig.colorbar(p)

    ax.set_xticks(np.arange(len(x_tick_labels)) + 0.5)
    ax.set_xticklabels(x_tick_labels)
    #pdb.set_trace()

    if len(y_tick_labels) == 0:
        ax.tick_params(labelleft='off')
    else:
        ax.set_yticks(np.arange(len(y_tick_labels)) * (len(data)/(len(y_tick_labels)-1)))
        ax.set_yticklabels(y_tick_labels)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(plot_title, fontsize=20)
    
    ax.invert_yaxis()
        
    #plt.colorbar()
    fig.tight_layout()
    plt.show()

#-------------------------------------------
def plotHistogramSubPlots(subplot_data, titles, x_labels, y_labels, x_tick_label_list, y_tick_label_list, rows, cols):

    if rows == 1 and cols == 1:
        plotHistogram(subplot_data[0], titles[0], x_labels[0], y_labels[0], x_tick_label_list[0], y_tick_label_list[0])
        return

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 12))

    plt.set_cmap('Reds')

    data_counter = 0

    for i in range(rows):

        for j in range(cols):

            data = subplot_data[data_counter]
            title = titles[data_counter]
            x_label = x_labels[data_counter]
            y_label = y_labels[data_counter]
            x_tick_labels = x_tick_label_list[data_counter]
            y_tick_labels = y_tick_label_list[data_counter]

            #p = ax.pcolormesh(data)

            if rows > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            ax.pcolor(data)

            ax.set_ylim(0,len(data))                                                                                                               
            ax.set_xlim(0,len(data[0]))
             
            #fig.colorbar(p)

            ax.set_xticks(np.arange(len(x_tick_labels)) + 0.5)
            ax.set_xticklabels(x_tick_labels)    

            if len(y_tick_labels) == 0:
                ax.tick_params(labelleft='off')
            else:
                ax.set_yticks(np.arange(len(y_tick_labels)) * (len(data)/(len(y_tick_labels)-1)))
                ax.set_yticklabels(y_tick_labels)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            ax.set_title(title, fontsize=12)
            
            ax.invert_yaxis()

            data_counter += 1

            if data_counter >= len(subplot_data):
                break

        if data_counter >= len(subplot_data):
            break
        
    #plt.colorbar()
    fig.tight_layout()
    plt.show()

def plotBarChartsSubPlots(subplot_data, titles, x_labels, rows=1, cols=1):

    if rows == 1 and cols == 1:
        plotBarCharts(subplot_data[0], titles[0], x_labels[0])
        return

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 12))

    data_counter = 0

    for i in range(rows):

        for j in range(cols):

            data = subplot_data[data_counter]
            title = titles[data_counter]
            x_label = x_labels[data_counter]

            if rows > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            ax.bar(range(len(x_label)), data, color="r", align="center")

            #pdb.set_trace()
            ax.set_xticks([k for k in range(len(x_label)) if k % 10 == 0])
            ax.set_xticklabels([x_label[k] for k in range(len(x_label)) if k % 10 == 0], rotation="vertical")
            ax.set_xlim([-1, len(x_label)])

            ax.set_title(title, fontsize=12)

            data_counter += 1

            if data_counter >= len(subplot_data):
                break

        if data_counter >= len(subplot_data):
            break

    fig.tight_layout()
    plt.show()


def plotBarCharts(data, title, x_label):

    plt.figure()
    plt.title(title)

    plt.bar(range(len(x_label)), data, color="r", align="center")
        
    plt.xticks(range(len(x_label)), x_label, rotation="vertical")
    plt.xlim([-1, len(x_label)])
    plt.tight_layout()
    plt.show()

def plotBarChartsSamePlot(data_container, title, x_label, multi_plot_legends):

    plt.figure()
    plt.title(title)
    
    x_range = np.asarray(range(len(x_label)))
    j = 0.2
    cmap = get_cmap(len(data_container))
    #colors = ["r", "g", "b", "y"]
    for i in range(len(data_container)):
        if len(data_container[i]) > 0:
            plt.bar(x_range+j*i, data_container[i], width=j, color=cmap(i), align="center", label=multi_plot_legends[i])
        
    plt.legend()
    plt.xticks(range(len(x_label)), x_label, rotation="vertical")
    plt.xlim([-1, len(x_label)])
    plt.tight_layout()
    plt.show()

def getHist(l, num_bins, minm, maxm):

    if len(l) == 0:
        return [], []

    if minm == None:
        minm=min(l)
        
    if maxm == None:
        maxm=max(l)

    if minm == maxm:
        num_bins = 1
        bin_size = 1
    else:
        bin_size = int(math.ceil(float(maxm-minm)/num_bins))

    hist_bin = [0] * (num_bins + 1)

    for value in l:
	
    	if value >= minm and value <= maxm:
	    hist_bin[int((value - minm)/bin_size)] += 1

    x_label = []
    data = []

    for i in range(len(hist_bin)):

        #print str(i*bin_size+minm) + "-" + str((i+1)*bin_size+minm), hist_bin[i]

        x_label.append(str(i*bin_size+minm) + "-" + str((i+1)*bin_size+minm))
        data.append(hist_bin[i])

    return data, x_label

def showMultiHist(ls, num_bins, minm_all=None, maxm_all=None, plot=True, titles=[], rows=0, cols=0, multi_plot=False, multi_plot_title="", multi_plot_legends=[]):

    x_labels = []
    subplot_data = []

    if multi_plot:
        merged_ls = []
        for l in ls:
            merged_ls += l
        if minm_all == None:
            minm_all = min(merged_ls)
        if maxm_all == None:
            maxm_all = max(merged_ls)

    for l in ls:
        
        minm = minm_all
        maxm = maxm_all

        data, x_label = getHist(l, num_bins, minm, maxm)

        x_labels.append(x_label)
        subplot_data.append(data)

    if plot:

        if not multi_plot:

            if len(titles) == 0:
                titles = ["Histogram bins" for i in range(subplot_data)]

            if cols == 0:
                cols = min([3, len(subplot_data)])
            if rows == 0:
                rows = int(math.ceil(float(len(subplot_data))/cols))
                
            plotBarChartsSubPlots(subplot_data, titles, x_labels, rows, cols)

        else:
            if len(multi_plot_legends) == 0:
                multi_plot_legends = ["legend" + str(i+1) for i in range(len(subplot_data))]
            plotBarChartsSamePlot(subplot_data, multi_plot_title, x_label, multi_plot_legends)


def showHist(l, num_bins, minm=None, maxm=None, plot=True, title="Histogram bins"):

    data, x_label = getHist(l, num_bins, minm, maxm)

    if plot:
        plotBarCharts(data, title, x_label)
        

def showPerc(counter, length, perc):

    perc_done = (counter+1)*100/length

    if perc_done >= perc:
        print int(perc_done/10)*10, "%"
        return int(perc_done/10)*10 + 10

    return perc

def comparePeaks(x, y):

    if x[0] > y[0]:
        return 1
    elif x[0] == y[0]:
        if int(x[1]) > int(y[1]):
            return 1
        elif int(x[1]) == int(y[1]):
            if int(x[2]) > int(y[2]):
                return 1
    return -1

def getPeakDistance(e, p):

    if "_" in e:
        e = e.split("_")

    if "_" in p:
        p = p.split("_")

    if e[0] == p[0]:

        # region1 is downstream of region2
        if int(e[1]) >= int(p[2]):
            return int(e[1]) - int(p[2])

        # region1 is upstream of region2
        if int(p[1]) >= int(e[2]):
            return int(p[1]) - int(e[2])

	# region1 is overlapped with region2
        return -1

    return -2

def getPeakStats(peaks, get_header=False, show_hist=False):

    if get_header:
        return ["Total peaks", "Unique peaks", "Overlapping peaks", "Peaks without gaps", "Peaks with gaps", "Peaks in different Chromosome", "Min length", "Max length", "Avg length", "Min distance", "Max distance", "Avg distance"]

    peaks = pd.DataFrame(peaks)
    total_peaks = len(peaks)
    peaks = peaks.drop_duplicates()
    peaks = peaks.values.tolist()
    peaks.sort(cmp=comparePeaks)

    lengths = []
    distances = []
    overlapping_peaks = []
    consecutive_peaks = []
    gaps = []
    diff_chrom = []

    for i in range(len(peaks)):
        
        lengths.append(abs(int(peaks[i][1])-int(peaks[i][2])))
        if i > 0:
            d = getPeakDistance(peaks[i], peaks[i-1])    
            
            if d == -2:
                diff_chrom.append([peaks[i-1], peaks[i]])
            elif d == -1:
                overlapping_peaks.append([peaks[i-1], peaks[i]])
            else:
                if d == 0:
                    consecutive_peaks.append([peaks[i-1], peaks[i]])
                elif d > 0:
                    gaps.append(d)
                distances.append(d)

    if show_hist:
        showHist(lengths, 100, title="Length of the peaks")
        showHist(distances, 100, title="distance of the peaks")

    return [total_peaks, len(peaks), len(overlapping_peaks), len(consecutive_peaks), len(gaps), len(diff_chrom)+1, min(lengths), max(lengths), sum(lengths)/len(lengths), min(distances), max(distances), sum(distances)/len(distances)]

def getOverlappedPeaks(start, end, peak_file, peak):

    while end > start:

        mid = (start + end)/2

        if peak[0] < peak_file[mid][0]:
            end = mid
        elif peak[0] > peak_file[mid][0]:
            start = mid + 1
        elif int(peak[2]) < int(peak_file[mid][1]):
            end = mid
        elif int(peak[1]) > int(peak_file[mid][2]):
            start = mid + 1
        else:

            matched_indices = []

            for i in range(mid+1, end):
                if peak_file[i][0] == peak[0] and overlapped(peak_file[i][1], peak_file[i][2], peak[1], peak[2]):
                    matched_indices.append(i)
                else:
                    break

            for i in range(mid, start-1, -1):
                if peak_file[i][0] == peak[0] and overlapped(peak_file[i][1], peak_file[i][2], peak[1], peak[2]):
                    matched_indices.append(i)
                else:
                    break
                
            return matched_indices

        #print start, end, mid

    return []

def searchFile(file_name, text, func=lambda x:x, compareText = lambda x,y: x-y):
    
    with(open(file_name)) as f:

        start = 0
        f.seek(0, 2)  # Seek to EOF.
        size = f.tell()
        end = size - 1
        
        while end > start:

            mid = (start + end) >> 1

            if mid > 0:
                f.seek(mid - 1)  # Just to figure out where our line starts.
                f.readline()  # Ignore current unfinished line, find our line.
                midf = f.tell()
            else:
                midf = 0
                f.seek(midf)

            file_text = f.readline().strip()
            file_text_part = func(file_text)
                        
            res = compareText(text, file_text_part)
            
            if res < 0:
                end = mid
            elif res > 0:
                start = mid + 1
            else:
                offset = len(file_text)
                results = [file_text] # Record mid
                
                while f.tell() < size: # Record all the matches after mid
                    file_text = f.readline().strip()
                    file_text_part = func(file_text)
                    if file_text_part != text:
                        break
                    results.append(file_text)
                    
                for i in range(midf-2, start-1, -1): # Record all the matches before mid
                    f.seek(i)
                    if i <= 0 or f.read(1) == "\n":
                        file_text = f.readline().strip()
                        file_text_part = func(file_text)
                        if file_text_part != text:
                            break
                        results = [file_text] + results
                    
                return results

    return []

# ------------------------------------
def getPerc(x, n):
    if n == 0:
        return 0
    return round(x*100./n, 2)

# ------------------------------------
def getAvg(a):
    if len(a) == 0:
        return 0
    return round(sum(a)/len(a), 2)

# ------------------------------------
def getMWU(a, b):
    if len(set(a+b)) == 1:
        return 1
    return mwu(a, b).pvalue
