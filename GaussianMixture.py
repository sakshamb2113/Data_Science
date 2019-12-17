import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as pca
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
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
    for i in [x for x in dataframe.columns if x !="Z_Scratch"]:
        data2[i] = (dataframe[i] - mini[i])/(maxi[i] - mini[i])
    data2.to_csv(path_to_file + '-Normalised.csv')

def standardize(dataframe):
    mean = dataframe.mean()
    std = dataframe.std()
    for i in [x for x in dataframe.columns if x !="Z_Scratch"]:
        dataframe[i]=(dataframe[i] - mean[i]) / std[i]
    dataframe.to_csv(path_to_file + "-Standardised.csv")


def dimensionality_reduction(data,k):
    scaler=pca(n_components=k)
    #print(data)
    scaler.fit(data)
    data=scaler.transform(data)
    #print(len(data[0]))
    return data

def mixture_gaussian(dataframe):
    df=dataframe.groupby("Z_Scratch")
    class0=df.get_group(0)
    class1=df.get_group(1)
    X_train0,X_test0=train_test_split(class0,test_size=0.3,random_state=42,shuffle=True)
    X_train1,X_test1=train_test_split(class1,test_size=0.3,random_state=42,shuffle=True)
    Y_train0=X_train0["Z_Scratch"]
    Y_train1=X_train1["Z_Scratch"]
    Y_test0=X_test0["Z_Scratch"]
    Y_test1=X_test1["Z_Scratch"]
    X_train0.drop(columns=["Z_Scratch"],inplace=True)
    X_train1.drop(columns=["Z_Scratch"],inplace=True)
    X_test0.drop(columns=["Z_Scratch"],inplace=True)
    X_test1.drop(columns=["Z_Scratch"],inplace=True)
    Y_test=pd.concat([Y_test0,Y_test1])
    X_test=pd.concat([X_test0,X_test1])
    X_train0,X_train1,X_test,Y_test
    a=[]
    c=[]
    for q in [2,4,8,16]:
        classifier=GaussianMixture(n_components=q,reg_covar=1e-5)
        classifier.fit(X_train0)
        weight0=classifier.weights_
        p0=classifier.predict_proba(X_test)
        classifier1=GaussianMixture(n_components=q,reg_covar=1e-5)
        classifier1.fit(X_train1)
        weight1=classifier1.weights_
        p1=classifier1.predict_proba(X_test)
        data_predict=[]
        for i in range(len(p0)):
            e0=0
            e1=0
            for j in range(len(weight0)):
                e0+=weight0[j]*p0[i][j]
                e1+=weight1[j]*p1[i][j]
            if e0>e1 :
                data_predict.append(0)
            else:
                data_predict.append(1)
        print("q =",q)
        print("accuracyscore",accuracy_score(data_predict,Y_test))
        print("confusionmatrix",confusion_matrix(data_predict,Y_test))
        a.append(accuracy_score(data_predict,Y_test))
    plt.plot([2,4,8,16],a)
    plt.show()
        #print(len(p0))
        #print(len(p1))
        #for i in range(len(value))
        

def PCA(dataframe):
    for i in range(1,len(dataframe.columns)):
        print("reduced dimensions to :",i)
        df=dataframe["Z_Scratch"]
        dataframe.drop(columns=["Z_Scratch"],inplace=True)
        data=dimensionality_reduction(dataframe,i)
        dataframe=pd.concat([dataframe,df],axis=1)
        data=pd.DataFrame(data)
        data1=pd.concat([data,df],axis=1)
        mixture_gaussian(data1)        
    
    #print(data)
        
        


"""def mixture_gaussian2(dataframe):
    df=dataframe.groupby("Z_Scratch")
    class0=df.get_group(0)
    class1=df.get_group(1)
    X_train0,X_test0=train_test_split(class0,test_size=0.3,random_state=42,shuffle=True)
    X_train1,X_test1=train_test_split(class1,test_size=0.3,random_state=42,shuffle=True)
    Y_train0=X_train0["Z_Scratch"]
    Y_train1=X_train1["Z_Scratch"]
    Y_test0=X_test0["Z_Scratch"]
    Y_test1=X_test1["Z_Scratch"]
    X_train0.drop(columns=["Z_Scratch"],inplace=True)
    X_train1.drop(columns=["Z_Scratch"],inplace=True)
    X_test0.drop(columns=["Z_Scratch"],inplace=True)
    X_test1.drop(columns=["Z_Scratch"],inplace=True)
    Y_test=pd.concat([Y_test0,Y_test1])
    X_test=pd.concat([X_test0,X_test1])
    X_train0,X_train1,X_test,Y_test
    a=[]
    c=[]
    for q in [2,4,8,16]:
        classifier=GaussianMixture(n_components=q,reg_covar=1e-5)
        classifier.fit(X_train0)
        weight0=classifier.weights_
        p0=classifier.predict_proba(X_test)
        classifier1=GaussianMixture(n_components=q,reg_covar=1e-5)
        classifier1.fit(X_train1)
        weight1=classifier1.weights_
        p1=classifier1.predict_proba(X_test)
        data_predict=[]
        data_predict=np.argmax(classifier.score_samples(X_test),classifier1.score_samples(X_test))
        print("q =",q)
        print("accuracyscore",accuracy_score(data_predict,Y_test))
        print("confusionmatrix",confusion_matrix(data_predict,Y_test))
        a.append(accuracy_score(data_predict,Y_test))
    plt.plot([2,4,8,16],a)
    plt.show()
"""    
    
def shuffle(function_parameters):
    pass


def traintestsplit(dataframe):
    #print(dataframe.loc[:,dataframe.columns!="Class"])
    X_train,X_test,Y_train,Y_test=train_test_split(dataframe.loc[:,dataframe.columns!="Z_Scratch"],dataframe["Z_Scratch"],test_size=0.3,random_state=42,shuffle=True)
    #X_train.append(Y_train)
    #X_test.append(Y_test)
    #train=pd.concat([X_train,Y_train])
    #test=pd.concat([X_test,Y_test])
    #print(Y_train)
    #print(Y_test)
    X_train.to_csv("/home/saksham/Downloads/SteelPlateFaults-2class-train.csv")
    X_test.to_csv("/home/saksham/Downloads/SteelPlateFaults-2class-test.csv")
    return X_train,Y_train,X_test,Y_test
#def confusion_matrix():
#    pass

def percentage_accuracy():
    pass

def kclassification(X_train,Y_train,X_test,Y_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    return knn.predict(X_test)

def bayesclassify(X_train,Y_train,X_test,Y_test):
    classifier=GaussianNB()
    classifier.fit(X_train,Y_train)
    data_predict=classifier.predict(X_test)
    return data_predict

def multimodalgaussian(X_train,Y_train,X_test,Y_test,k):
    classifier=GaussianMixture(n_components=k)
    classifier.fit(X_train)
    return classifier.predict(X_test)

path_to_file="/home/saksham/Downloads/SteelPlateFaults-2class"
data=load_dataset(path_to_file + '.csv')
PCA(data)
for i in range(1,len(data.columns)):
    #reduced_data=dimensionality_reduction(data,i)
    #print(reduced_data)
    #print(len(reduced_data.columns))
    print("reduced dimensions to",i)
    X_train,Y_train,X_test,Y_test=traintestsplit(data)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    reduced_X_train=dimensionality_reduction(X_train,i)
    reduced_X_test=dimensionality_reduction(X_test,i)
    #data_predict=kclassification(reduced_X_train,Y_train.values,reduced_X_test,Y_test.values,i)
    a=[]
    mx=0
    temp=0
    i=0
    c=[]
    for k in range(1,22,2):
        data_predict=kclassification(reduced_X_train,Y_train.values,reduced_X_test,Y_test.values,k)
        c.append(confusion_matrix(Y_test.values,data_predict))
        a.append(accuracy_score(Y_test.values,data_predict))
        if a[i]>mx:
            temp=k
            mx=a[i]
        i+=1
    print("k with maximum accuracy:",temp)
    plt.plot(range(1,22,2),a)
    plt.show()
   
    
