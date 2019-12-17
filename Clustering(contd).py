#import sklearn.cluster.AgglomerativeClustering as agglc
#import sklearn.cluster.DBSCAN as dbscan
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering as agglc
from sklearn.cluster import DBSCAN as dbscan
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
#from yellowbrick.cluster import KElbowVisualiser
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


data=pd.read_csv("/home/saksham/Downloads/Iris.csv")
sdata=data["Species"]
#print(sdata)
#y_true=data["Species"]
data=data.drop(columns=["Species"])
#digits=load_digits()
#y_true=digits.target
#print(np.shape(digits.data))
y_true=[]
for i in range(len(data)):
    if sdata[i]=="Iris-setosa":
        y_true.append(0)
    elif sdata[i]=="Iris-versicolor":
        y_true.append(1)
    else:
        y_true.append(2)
scaler=MinMaxScaler()
data = scaler.fit_transform(data)
#pca = PCA(n_components=10).fit(data)
reduced_data = PCA(n_components=2).fit_transform(data)
distk=[]
distg=[]
LABEL_COLOR_MAP = {
                       -1 : 'm',
                       0 : 'r',
                       1 : 'k',
                       2 : 'b',
                       3 : 'g',
                       4 : 'y',
                       5 : 'c',
                       6 : 'w',
                       7 : 'm',
                       8 : 'b',
                       9 : 'g',
                       10 : 'r',
                       11 : 'k',
                       12 : 'b',
                       13 : 'g',
                       14 : 'y',
                       15 : 'c',
                       16 : 'w',
                       17 : 'm',
                       18 : 'b',
                       19 : 'g',
                       20 : 'r',
                       21 : 'k',
                       22 : 'b',
                       23 : 'g',
                       24 : 'y',
                       25 : 'c',
                       26 : 'w',
                       27 : 'm',
                       28 : 'b',
                       29 : 'g',
                       30 : 'r',
                      }
for k in [2,3,4,5,6,7]:
    kmeans=KMeans(n_clusters=k,random_state=0).fit(reduced_data)
    print("kmeans\nno of clusters",k)
    labels=kmeans.labels_
    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
    plt.show()
    sum=list(range(k))
    centers=kmeans.cluster_centers_
    #for i in (range(len(kmeans.labels_))):
    #    sum[labels[i]]+=(reduced_data[i][0]-centers[labels[i]][0])**2+(reduced_data[i][1]-centers[labels[i]][1])**2
    sum=kmeans.inertia_
    print(sum)
    #print(labels)
    distk.append(sum/k)
    print("purity score !\n",purity_score(y_true,labels))

    
    
    
print("agglomerative clustering")
hc=agglc(n_clusters=3).fit(reduced_data)
labels=hc.labels_
label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
plt.show()
print("purity score !\n",purity_score(y_true,labels))

print("dbscan")
db=dbscan().fit(reduced_data)
labels=db.labels_
print("default parameters")
label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
plt.show()
print("purity score !\n",purity_score(y_true,labels))




for e in [0.05,0.5,0.95]:
    db=dbscan(eps=e).fit(reduced_data)
    labels=db.labels_
    #print(labels)
    print("\n\nmaximum distance between two samples =",e)
    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
    plt.show()
    print("purity score !\n",purity_score(y_true,labels))





for s in [1,5,10,20]:
    db=dbscan(eps=0.05,min_samples=s).fit(reduced_data)
    labels=db.labels_
    #print(labels)
    print("\n\nminpts =",s)
    label_color = cm.rainbow(np.linspace(0,1,len(labels)))
    #label_color = itertools.cycle(["r", "b", "g","y","c","w","m","k"])
    #print(set(labels))
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
    plt.show()
    print("purity score !\n",purity_score(y_true,labels))



#KElbowVisualiser
