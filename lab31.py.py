import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(path_to_file):
    data=pd.read_csv(path_to_file)
    return data

def show_box_plot(attribute_name,dataframe):
    print("boxplot for",attribute_name)
    plt.boxplot(dataframe[attribute_name])
    plt.show()
    
def replace_outliers(dataframe):
    #dcols=[x for x in dataframe.columns if x not in ["dates","stationid"]]
    #print(dcols)
    for col in [x for x in ["temperature","humidity","rain"]]:
        #print(dataframe.col)
        q1,q3=np.percentile(dataframe[col],[25,75])
        med=dataframe[col].median()
        #print(med)
        iqr=q3-q1
        #print(q3+1.5*iqr)
        #print(q1-1.5*iqr)
        #print(dataframe.loc[dataframe[col]<q1-1.5*iqr,col])
        dataframe.loc[dataframe[col]>q3+1.5*iqr,col]=np.nan
        dataframe.loc[dataframe[col]<q1-1.5*iqr,col]=np.nan
        #print(dataframe.loc[dataframe[col]<q1-1.5*iqr,col])
        #print(dataframe)
        dataframe[col]=dataframe[col].fillna(med)
        #print(dataframe.loc[412,"humidity"])
        plt.boxplot(dataframe[col])
        plt.show()
    return dataframe

#def range(a):
    #pass

def min_max_normalization(dataframe,s):
    data=dataframe.copy()
    for col in [x for x in ["temperature","humidity","rain"]]:
        a=[dataframe[col].min(),dataframe[col].max()]
        #print(range(5))
        for i in range(len(dataframe)):
            data[col][i]=s*(dataframe[col][i]-a[0])/(a[1]-a[0])
    return data


def standardize(dataframe):
    for col in [x for x in ["temperature","humidity","rain"]]:
        mu=dataframe[col].mean()
        stdev=dataframe[col].std()
        for i in range(len(dataframe)):
            dataframe[col][i]=(dataframe[col][i]-mu)/stdev
        return dataframe



path_to_file="/home/saksham/Downloads/landslide_data2_original.csv"
dataframe=read_data(path_to_file)
show_box_plot("temperature",dataframe)
show_box_plot("humidity",dataframe)
show_box_plot("rain",dataframe)
datanew=replace_outliers(dataframe)
minmaxdata=min_max_normalization(datanew,1)
#show_box_plot("temperature",minmaxdata)
#show_box_plot("humidity",minmaxdata)
#show_box_plot("rain",minmaxdata)
minmaxdata=min_max_normalization(datanew,20)
#show_box_plot("temperature",minmaxdata)
#show_box_plot("humidity",minmaxdata)
#show_box_plot("rain",minmaxdata)
datastandard=standardize(datanew)


