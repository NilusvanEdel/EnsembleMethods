import numpy as np 
import csv
from scipy.misc import imread, imresize


class Batch():
    '''
    Deals samples in batches
    '''
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def next(self, size):
        idx = np.random.permutation(self.X.shape[0])[:size]
        x = self.X[idx,:]
        y = self.Y[idx]

        #self.X = self.X[size:,:]
        #self.Y = self.Y[size:]

        return x,y 

def read_data(f_name = 'titanic_full.csv', 
              adapt_titles = False, 
              adapt_ages = False,
              adapt_price = False):
    file = open(f_name)
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)
    data = data[:,1:]
    Y = data[1:,0]
    codeing = np.unique(Y)
    Y = ((Y==codeing[0])+0.)
    X = data[1:,1:]
    feature_names = data[0,:]
    #X[X == ''] = '-1'
    '''
    ### Mine Surnames and Titles
    '''
    n_lengths = np.array([len(x[1]) for x in X])
    surnames = np.array([x[1].split()[0] for x in X])
    titles = np.array([x[1].split(',')[1].split('.')[0].strip() for x in X])
    if adapt_titles:
        # substitute rare titles
        rare_idx = [ix for ix,t in enumerate(titles) if not t in ['Mr','Miss','Mrs','Master']]
        titles[rare_idx] = 'Rare_Title'
    has_parentheses = np.array([x[1].count('(') for x in X])
    X[:,1] = surnames
    X = np.vstack((X.T,n_lengths,titles,has_parentheses)).T
    """
    ### substitute missing age by mean age of title
    """
    if adapt_ages:
        ut = np.unique(titles)
        mean_ages = []
        for t in ut:
            ages = X[titles == t,3]
            ages = [float(a) for a in ages if not a == '']
            mean_ages.append(np.mean(ages))
        mean_ages = np.array(mean_ages)
        for idx in np.where(X[:,3] == '')[0]:
            title = X[idx,-2]
            X[idx,3] = mean_ages[ut==title][0]
    """
    ### estimate missing ticket prices from pclass
    """
    if adapt_price:
        u_classes = np.unique(X[:,0])
        mean_per_class = []
        for c in u_classes:
            idx = X[:,0] == c
            fares = [float(x) for x in X[idx,7] if not x=='']
            mean_per_class.append(np.mean(fares))
        """
        ### substitute missing fare prices based on class
        """
        for px in range(X.shape[0]):
            if X[px,7] == '':
                cl = X[px,0]
                X[px,7] = mean_per_class[int(cl)-1]
    """
    ### split-up ticket number into letters and numbers
    """
    tickets = []
    for t in X[:,6]:
        tmp = t.split(' ')
        if len(tmp) < 2:
            if not tmp[0] == 'LINE':
                tmp = ['',tmp[0]]
            else:
                tmp = ['LINE', '0000']
        elif len(tmp) > 2:
            tmp = [tmp[0]+' '+tmp[1],tmp[2]]
        tickets.append(tmp)
    tickets = np.array(tickets)
    # delete ticket from X
    idx = np.invert(np.arange(0,X.shape[1],1)==6)
    X = X[:,idx]
    X = np.hstack((X,tickets))
    idx = feature_names != 'Ticket'
    feature_names = feature_names[idx]
    feature_names = np.hstack((feature_names,['length_of_name','Title','parentheses','Ticket_letter','Ticket_number']))[1:]
    feature_names[1] = 'Surname'
    # shuffle data
    idx = np.random.permutation(Y.shape[0])
    Y = Y[idx]
    X = X[idx,:]
    file.close()
    return X, Y, feature_names,codeing

def read_image(f_name = 'test.png'):
    data = imread(f_name)
    if len(data.shape)>2:
        data = data[:,:,0]
    data = np.array([(data[i,j],i,j) for i in range(data.shape[0]) for j in range(data.shape[1])])
    X = data[:,1:]
    Y = data[:,0]
    vals = np.unique(Y)
    codeing = np.arange(0,len(vals))
    for ix,v in enumerate(vals):
        Y[Y==v] = codeing[ix]
    Y = Y*2-1
    #idx = np.random.permutation(Y.shape[0])
    #Y=Y[idx]
    #X=X[idx,:]
    return X,Y


def split_sets(X,Y,ratio_train=.4,ratio_test=.2):
    idx = np.random.permutation(X.shape[0])
    X = X[idx,:]
    Y = Y[idx]
    l = Y.shape[0]
    X_train = X[:int(l*ratio_train),:]
    X = X[int(l*ratio_train):,:]
    Y_train = Y[:int(l*ratio_train)]
    Y = Y[int(l*ratio_train):]
    X_test = X[:int(l*ratio_test),:]
    X = X[int(l*ratio_test):,:]
    Y_test = Y[:int(l*ratio_test)]
    Y = Y[int(l*ratio_test):]
    X_vali = X
    Y_vali = Y

    return X_train,Y_train,X_test,Y_test,X_vali,Y_vali
