from statistics import *
import csv 
def find_k_best(fname, k=20):
    features = []
    res_avg = []
    res_dt = []
    res_rfw = []

    f = open(fname)
    for i, line in enumerate(f):
        rdt, rrfw, feature = line.split('\t')
        res_dt.append((float(rdt), i))
        res_rfw.append((float(rrfw), i))
        res_avg.append(((float(rdt) + float(rrfw))/2, i))
        features.append(feature.split(','))

    res_avg = sorted(res_avg, reverse=True)
    res_dt = sorted(res_dt, reverse=True)
    res_rfw = sorted(res_rfw, reverse=True)

    arr = [['Average AUC', 'Average AUC Features', 'DT AUC', 'DT AUC Features', 'RFW AUC', 'RFW AUC Features']]
    
    for i in range(k):
        arr.append([
            res_avg[i][0], features[res_avg[i][1]],
            res_dt[i][0], features[res_dt[i][1]],
            res_rfw[i][0], features[res_rfw[i][1]]
                ])
    
    # write to csv file
    # 10 fold and leave one out analysis on chsoen sample points TODO
    f = open(fname + 'parsed.csv', 'w', newline='')
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    writer.writerows(arr)

find_k_best('results7', 100)

def find_stats(arr):
    pass 
    return mean, std, var
    
def commonfeatures_k_best(fname):
    pass 
    

    