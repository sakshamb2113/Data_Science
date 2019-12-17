import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
#from yellowbrick.cluster import KElbowVisualiser
from sklearn import metrics

def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
 # return purity
 return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix)

data=pd.read_csv(r"C:\Users\saksh\Downloads\Iris.csv")
print(data)
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
for k in [2,3,4,5,6,7]:
    kmeans=KMeans(n_clusters=k,random_state=0).fit(reduced_data)
    print("kmeans")
    labels=kmeans.labels_
    LABEL_COLOR_MAP = {0 : 'r',
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
                       
                      }
    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color)
    plt.show()
    sum=list(range(k))
    centers=kmeans.cluster_centers_
    #for i in (range(len(kmeans.labels_))):
    #    sum[labels[i]]+=(reduced_data[i][0]-centers[labels[i]][0])**2+(reduced_data[i][1]-centers[labels[i]][1])**2
    sum=kmeans.inertia_
    print(sum/k)
    #print(labels)
    distk.append(np.average(sum))
    print("gmm")
    gmm=GaussianMixture(n_components=k)
    gmm.fit(reduced_data)
    data_predict=gmm.predict(reduced_data)
    label_color2=[LABEL_COLOR_MAP[l] for l in data_predict]
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color2)
    plt.show()
    print(gmm.score(reduced_data))
    distg.append(gmm.score(reduced_data))
    acc=purity_score(y_true,data_predict)
    print(acc)
print(distg)
#KElbowVisualiser
plt.plot([2,3,4,5,6,7],distk)
plt.show()
plt.plot([2,3,4,5,6,7],distg)
plt.show()
