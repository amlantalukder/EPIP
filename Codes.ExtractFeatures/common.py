import os

# -----------------------------------------------
def writeFile(filename, data, mode="w"):
    d = os.path.dirname(filename)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

    fl = open(filename, mode)
    fl.write(data)
    fl.close()

# -----------------------------------------------
def formatDataTable(data, col_sep="\t", row_sep="\n"):
    return row_sep.join([col_sep.join([str(item1) for item1 in item]) for item in data])

# -----------------------------------------------
def writeDataTableAsText(data, filename, mode="w"):
    text = formatDataTable(data, "\t", "\n")

    writeFile(filename, text, mode)

# -----------------------------------------------
def writeDataTableAsCSV(data, filename, mode="w"):
    text = formatDataTable(data, ",", "\n")

    writeFile(filename, text, mode)

# -----------------------------------------------
def readFileInTable(filename, delim='\t'):

    fl = open(filename, "r")
    data = fl.readlines()
    fl.close()
    data = [item.strip().split(delim) for item in data]
    return data

# -----------------------------------------------
def showPerc(counter, length, perc, perc_inc=10):
    perc_done = (counter * 100) // length

    if perc_done >= perc:
        print('{}%'.format(int(perc_done / 10) * 10))
        return int(perc_done / 10) * 10 + perc_inc

    return perc

# -----------------------------------------------
def overlapped(start1, end1, start2, end2):
    start1, end1, start2, end2 = int(start1), int(end1), int(start2), int(end2)

    if (start1 <= start2 and end1 >= start2) or \
            (start2 <= start1 and end2 >= start1):
        return True

    return False

# -----------------------------------------------
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

# ------------------------------------------------------------------------------------------------
def checkPeakOverlap(peak_file_sorted_start, peak_file_sorted_end, peak, nn_allowed=False):
    def bin_search(peak_file, peak, eq_ind=0):

        start = 0
        end = len(peak_file)

        while end > start:

            mid = (start + end) // 2
            if peak[0] < peak_file[mid][0]:
                end = mid
            elif peak[0] > peak_file[mid][0]:
                start = mid + 1
            elif peak[1] < peak_file[mid][1]:
                end = mid
            elif peak[1] > peak_file[mid][1]:
                start = mid + 1
            elif peak[1] == peak_file[mid][1]:
                if eq_ind == 1:
                    end = mid
                else:
                    start = mid + 1

            else:
                return mid

        return end

    # ------------------------------------------------------------------------------------------------
    # Find all peaks in the database that start after the query peak ends
    # ------------------------------------------------------------------------------------------------
    i = bin_search(peak_file_sorted_start, [peak[0], int(peak[2])])

    # ------------------------------------------------------------------------------------------------
    # i is the number of peaks that are candidate for overlap. If it is zero, there is no overlapping
    # peaks in the database with the query peak
    # ------------------------------------------------------------------------------------------------
    if i == 0 and not nn_allowed: return []

    # ------------------------------------------------------------------------------------------------
    # Find all peaks in the database that end before the query peak start
    # ------------------------------------------------------------------------------------------------
    j = bin_search(peak_file_sorted_end, [peak[0], int(peak[1])], 1)

    # ------------------------------------------------------------------------------------------------
    # j is the number of peaks that are not candidate for overlap. If it is equal to the total length,
    # there is no overlapping peaks in the database with the query peak
    # ------------------------------------------------------------------------------------------------
    if j == len(peak_file_sorted_end) and not nn_allowed: return []

    # ------------------------------------------------------------------------------------------------
    # If i > j then, there are (i-j) overlapping peaks.
    # ------------------------------------------------------------------------------------------------
    if i > j:
        candidate_indices1 = set()
        for k in range(i - 1, -1, -1):
            if peak_file_sorted_start[k][0] != peak[0]:
                break
            candidate_indices1.add(peak_file_sorted_start[k][-1])

        candidate_indices2 = set()
        for k in range(j, len(peak_file_sorted_end)):
            if peak_file_sorted_end[k][0] != peak[0]:
                break
            candidate_indices2.add(peak_file_sorted_end[k][-1])

        return list(candidate_indices1.intersection(candidate_indices2))

    # ------------------------------------------------------------------------------------------------
    # Nearest neighbour (NN) can be the overlapping peaks. If there are no overlapping peaks, we
    # return the indices of the peak that starts after the query peak ends (i) and the peak that ends
    # before the query peak starts (j-1)
    # ------------------------------------------------------------------------------------------------
    if nn_allowed:
        if i < len(peak_file_sorted_start) and j > 0:
            return [peak_file_sorted_end[j - 1][-1], peak_file_sorted_start[i][-1]]
        elif i >= len(peak_file_sorted_start):
            return [peak_file_sorted_end[j - 1][-1]]
        elif j <= 0:
            return [peak_file_sorted_start[i][-1]]

    # ------------------------------------------------------------------------------------------------
    # if i-j <= 0, then there is no overlapping peaks
    # ------------------------------------------------------------------------------------------------
    return []

# ------------------------------------------------------------------------------------------------
# For each peak in peak_file1, search in peak_file2
# ------------------------------------------------------------------------------------------------
def getOverlappedPeaks2(peak_file1, peak_file2, nn_allowed=False):

    peak_file2_sorted_start = []
    peak_file2_sorted_end = []

    for i in range(len(peak_file2)):
        peak_file2_sorted_start.append([peak_file2[i][0], int(peak_file2[i][1]), i])
        peak_file2_sorted_end.append([peak_file2[i][0], int(peak_file2[i][2]), i])

    peak_file2_sorted_start = sorted(peak_file2_sorted_start)
    peak_file2_sorted_end = sorted(peak_file2_sorted_end)

    overlapped_indices = {}
    for i in range(len(peak_file1)):
        indices = checkPeakOverlap(peak_file2_sorted_start, peak_file2_sorted_end, peak_file1[i], nn_allowed)
        if len(indices) > 0: overlapped_indices[i] = indices

    return overlapped_indices