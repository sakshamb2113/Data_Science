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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.graphics.tsaplots as stplot
import statsmodels 
import operator
import datetime
from mpl_toolkits.mplot3d import Axes3D

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

def traintestsplit(dataframe):
    #print(dataframe.loc[:,dataframe.columns!="Class"])
    X_train,X_test,Y_train,Y_test=train_test_split(dataframe.loc[:,dataframe.columns!="temperature"],dataframe["temperature"],test_size=0.5,random_state=42,shuffle=True)
    #X_train.append(Y_train)
    #X_test.append(Y_test)
    #train=pd.concat([X_train,Y_train])quality
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

path_to_file="C:/Users/saksh/Downloads/SoilForce"
data=load_dataset(path_to_file + '.csv')
#dates=matplotlib.dates.datestr2num(data["Date"])
converted_dates = list(map(datetime.datetime.strptime, data["Date"], len(data["Date"])*['%d-%m-%Y']))
formatter=matplotlib.dates.DateFormatter('%d-%m-%Y')
plt.plot(converted_dates,data["Force"])
ax=plt.gcf().axes[0]
ax.xaxis.set_major_formatter(formatter)
plt.gcf().autofmt_xdate(rotation=25)
plt.show()

#lag_table=pd.concat([data["Rain(mm)"][0:len(data["Rain(mm)"])-1],data["Rain(mm)"][1:len(data["Rain(mm)"])]],axis=1)
temp=data["Force"].shift(periods=1)
lag_table=pd.DataFrame({"Force":temp,"real":data["Force"][1:len(data["Force"])]})
lag_table.drop(0,inplace=True)
for i in range(1,30):
    temp=temp.shift(periods=1)
    lag_table=pd.concat([temp,lag_table],axis=1)
    lag_table.drop(i,inplace=True)
    temp=lag_table.iloc[:,0]
lag_table.drop(0,inplace=True)
#print(lag_table)
co_mat=lag_table.corr(method='pearson')['real']
#print(co_mat[-2::-1])
plt.plot(range(1,31),lag_table.corr(method='pearson')['real'][-2::-1],marker="<")
plt.show()
#print(X_train)
X_train,X_test=train_test_split(data,test_size=0.5,shuffle=False)
converted_train_dates = list(map(datetime.datetime.strptime,X_train["Date"], len(X_train["Date"])*['%d-%m-%Y']))
#print(X_test)
#stplot.plot_acf()
#print(lag_table)
#print(lag_table.iloc[:,0],lag_table.iloc[:,1]) 
persist=pd.DataFrame({"lag":data["Force"].shift(1),"real":data["Force"]})
persist.drop(0,inplace=True)
#print(persist)
b=int(len(persist)/2)
train,test=persist[1:b],persist[b:]
print("rmse for persistence model :",(mse(test["lag"],test["real"]))**0.5)
mymodel=statsmodels.tsa.ar_model.AR(X_train["Force"],dates=converted_train_dates)
my_fit=mymodel.fit()
predictions = my_fit.predict(start=len(X_train), end=len(X_train)+len(X_test)-1, dynamic=False)
print("rmse for autoregression model :",mse(predictions,X_test["Force"])**0.5)
print("optimal lag :",my_fit.k_ar)
print("autoregression coefficients :",my_fit.params)