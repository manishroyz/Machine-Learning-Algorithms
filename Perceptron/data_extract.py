import numpy as np

def extract(file_names):
    data = []
    labels = []
    
    for file_name in file_names:
        with open(file_name) as f:
            for i in f.readlines():
                data.append(i.strip())
            
    for i,x in enumerate(data):
        if (x.split(' ')[0] == '+1'):
            labels.append(1)
        else:
            labels.append(-1)
        data[i] = x.split(' ')[1:]
    
    data = np.array(data)
    feat_count = 19
    final_data = []
    
    for x in data:
        t = [0]*feat_count
        for y in x:
            p = y.split(':')
            t[int(p[0]) - 1] = float(p[1])
        final_data.append(t)

    pos = 0
    neg = 0
    for i in labels:
        if i == 1:
            pos += 1
        else :
            neg += 1
    
    final_data = np.array(final_data)
    return np.c_[final_data, labels]