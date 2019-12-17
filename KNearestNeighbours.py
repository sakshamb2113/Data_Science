import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def load_dataset(path_to_file):
    data=pd.read_csv(path_to_file)
    return data


def outliers_detection():
    pass    

def missing_values(dataframe):
    dataframe.fillna(dataframe.median())

def min_max_normalization(dataframe):
    data2 = dataframe.copy()
    mini = dataframe.min()
    maxi = dataframe.max()
    for i in [x for x in dataframe.columns if x !="Class"]:
        data2[i] = (dataframe[i] - mini[i])/(maxi[i] - mini[i])
    data2.to_csv(path_to_file + '-Normalised.csv')

def standardize(dataframe):
    mean = dataframe.mean()
    std = dataframe.std()
    for i in [x for x in dataframe.columns if x !="Class"]:
        dataframe[i]=(dataframe[i] - mean[i]) / std[i]
    dataframe.to_csv(path_to_file + "-Standardised.csv")


def dimensionality_reduction():
    pass


def shuffle(function_parameters):
    pass


def traintestsplit(dataframe):
    #print(dataframe.loc[:,dataframe.columns!="Class"])
    X_train,X_test,Y_train,Y_test=train_test_split(dataframe.loc[:,dataframe.columns!="Class"],dataframe["Class"],test_size=0.3,random_state=42,shuffle=True)
    #X_train.append(Y_train)
    #X_test.append(Y_test)
    #train=pd.concat([X_train,Y_train])
    #test=pd.concat([X_test,Y_test])
    #print(Y_train)
    #print(Y_test)
    X_train.to_csv("/home/saksham/Downloads/DiabeticRetinipathy-train.csv")
    X_test.to_csv("/home/saksham/Downloads/DiabeticRetinipathy-test.csv")
    return X_train,Y_train,X_test,Y_test
#def confusion_matrix():
#    pass

def percentage_accuracy():
    pass

def classification(X_train,Y_train,X_test,Y_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    return knn.predict(X_test)
    
path_to_file="/home/saksham/Downloads/DiabeticRetinipathy"
data=load_dataset(path_to_file + '.csv')
#print(data)
min_max_normalization(data)
#print(data)
standardize(data)
#print(data)
data_normal=load_dataset(path_to_file + '-Normalised.csv')
print(data_normal.columns)
data_standard=load_dataset(path_to_file + '-Standardised.csv')
missing_values(data)
missing_values(data_normal)
missing_values(data_standard)
X_train,Y_train,X_test,Y_test=traintestsplit(data)

a=[]
mx=0
temp=0
i=0
c=[]
for k in range(3,22,2):
    data_predict=classification(X_train,Y_train,X_test,Y_test,k)
    c.append(confusion_matrix(Y_test.values,data_predict))
    a.append(accuracy_score(Y_test.values,data_predict))
    if a[i]>mx:
        temp=k
        mx=a[i]
    i+=1
print("k with maximum accuracy:",temp)
plt.plot(range(3,22,2),a)    
#print(c[1])
            
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
a=[]
mx=0
temp=0
i=0
c=[]
for k in range(3,22,2):
    y_predict=classification(X_train,Y_train,X_test,Y_test,k)
    c.append(confusion_matrix(Y_test.values,y_predict))
    a.append(accuracy_score(Y_test.values,y_predict))
    if a[i]>mx:
        temp=k
        mx=a[i]
    i+=1
print("k with maximum accuracy:",temp)
plt.plot(range(3,22,2),a)
#print(c[1])


#X_train,Y_train,X_test,Y_test=traintestsplit(data_standard)
#print(X_train.std())
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
a=[]
mx=0
temp=0
i=0
c=[]
for k in range(3,22,2):
    data_predict=classification(X_train,Y_train,X_test,Y_test,k)
    c.append(confusion_matrix(Y_test.values,data_predict))
    a.append(accuracy_score(Y_test.values,data_predict))
    if a[i]>mx:
        temp=k
        mx=a[i]
    i+=1
print("k with maximum accuracy:",temp)
plt.plot(range(3,22,2),a)
plt.show()
#print(c[1])  
