import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as pca
from sklearn.naive_bayes import GaussianNB
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


def dimensionality_reduction(data,k):
    scaler=pca(n_components=k)
    #print(data)
    scaler.fit(data)
    data=scaler.transform(data)
    #print(len(data[0]))
    return data

    


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

def kclassification(X_train,Y_train,X_test,Y_test,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    return knn.predict(X_test)

def bayesclassify(X_train,Y_train,X_test,Y_test):
    classifier=GaussianNB()
    classifier.fit(X_train,Y_train)
    data_predict=classifier.predict(X_test)
    return data_predict

path_to_file="/home/saksham/Downloads/DiabeticRetinipathy"
data=load_dataset(path_to_file + '.csv')
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
    
    
    mx=0
    temp=0
    i=0
    data_predict=bayesclassify(reduced_X_train,Y_train.values,reduced_X_test,Y_test.values)
    print("confusion matrix for bayes",confusion_matrix(Y_test.values,data_predict))
    print("accuracy score for bayes",accuracy_score(Y_test.values,data_predict))
    
    #print("k with maximum accuracy:",temp)

