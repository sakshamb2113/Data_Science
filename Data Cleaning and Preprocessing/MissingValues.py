import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
missing_values = ["n/a", "na", "--"]
data=pd.read_csv("/home/saksham/Downloads/winequality-red_miss.csv",na_values=missing_values)
datao=pd.read_csv("/home/saksham/Downloads/winequality-red_miss (copy).csv",na_values=missing_values)
origin=pd.read_csv("/home/saksham/Downloads/winequality-red_original.csv",sep=";",na_values=missing_values)
cnt=0
#print(data)
#print(type(data["volatile acidity"][27]))
r=[]
n=[]
d=data.isnull()
g=data.isnull()
#print(d)
cnt50percent=0
for i in range(len(data)):
    cnt=0
    for col in data.columns:
        if d[col][i]==True :
            cnt+=1
            
    if cnt!=0:
        r.append(cnt)
    if cnt>=6:
        n.append(i)
        cnt50percent+=1
r=np.unique(r,return_counts=True)
plt.plot(r[0],r[1])
plt.show()
print("q2",cnt50percent)
print(len(data))
data.drop(n,inplace=True)
d=data.isnull()

#print(list(range(len(d)))
#-n)])**2
new=new/64
for i in [x for x in range(len(d)) if x not in n]:
    if d["quality"][i]==True:
        n.append(i)
        data=data.drop([i])
print(len(data))
d=data.isnull()
print(data.isna().sum())
print(64)
new=0
old=[]
print(datao)
for col in data.columns:
    print(col)
    new=0
    datao[col]=datao[col].fillna(data[col].median())
    for i in [x for x in range(len(g)) if g[col][x]==True]:
        new+=(datao[col][i]-origin[col][i])**2
    new=new/64
    print("rms: ",new**0.5)
#print(data)
k=datao.values-origin.values
#print(k)
#print(data["citric acid"][0])
for col in data.columns:
    print("modified     original")
    print(col)
    print("mean")
    print(data[col].mean(),datao[col].mean())
    print("median")
    print(data[col].median(),datao[col].median())
    print("mode")
    print(data[col].mode(),datao[col].mode())
    print()
#print("rms: ",new**0.5)
    


