class Band:
    def __init__(self,sell,level,overall):
        self.s=sell#唱片销量
        self.l=level#乐队等级
        self.o=overall#总产箱

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LinearRegression
rank_thershold=[0,9,11,13,15,17]#决定演出等级的阈值
def concert(band, place):
    base=[75,200,75,250,100,150]#各演出等级的产箱因子
    dice=random.randint(1,6)+random.randint(1,6)+random.randint(1,6)#3d6+演出等级，取这个值大于的最高阈值作为演出等级
    disband=0
    promote=0
    for i in range(6):
        if (dice+band.l)>rank_thershold[i]:
            rank=i
        else:
            break
    band.o+=(band.s+base[rank])*place/100
    if rank<2:
        disband=1
    else:
        if rank>3:
            promote=1        
        band.s+=50*(rank-1)
    return band,disband,promote

def run (practicetimes, days,trait1,trait2):
    band=Band(0,1,0)#直接在这里修改
    promote=0
    disband=0
    concertcount=0
    for i in range(days):
        if promote==1:
            if band.l<=4:
                band.l+=1
                promote=0
                continue
        if disband==1:
            break
        #if concertcount==practicetimes:
         #   band.l-=2#还有这里
        if concertcount<practicetimes:
            band.l+=trait1#特殊升级
            band,disband,promote=concert(band,250)#和这里
            band.l-=trait1
        else:
            band.l+=trait2
            band,disband,promote=concert(band,1000)
            band.l-=trait2
        concertcount+=1
    return band.o

def trail(n,practicetimes,days,trait1,trait2):
    res=[]
    for i in range(n):
        res.append(run(practicetimes,days,trait1,trait2))
    re=np.array(res)
    ave=np.average(re)
    std=np.std(re)
    return ave,std

def trial_c(n,practicetimes,days):
    res=[]
    for i in range(n):
        res.append(run(practicetimes,days))
    re=np.array(res)
    ave=np.average(re)
    std=np.std(re)
    return res,ave,std



fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)
ax1.set_yscale('log', basey=10)
ax2.set_yscale('log', basey=10)
ax3.set_yscale('log', basey=10)
ax4.set_yscale('log', basey=10)
n, bins, histpatches = ax1.hist(res1, 50, facecolor='green', alpha=0.75)
n, bins, histpatches = ax2.hist(res2, 50, facecolor='green', alpha=0.75)
n, bins, histpatches = ax3.hist(res3, 50, facecolor='green', alpha=0.75)
n, bins, histpatches = ax4.hist(res4, 50, facecolor='green', alpha=0.75)

def run (practicetimes, counts,sell,level):
    band=Band(sell,level,0)
    promote=0
    disband=0
    concertcount=0
    while concertcount<counts:
        if promote==1:
            band.l+=1
            promote=0
            continue
        if disband==1:
            break
        if concertcount==practicetimes:
            band.l-=0
        if concertcount<practicetimes:
            band,disband,promote=concert(band,1000)
        else:
            band,disband,promote=concert(band,500)
        concertcount+=1
    return band.o

def trail(n,practicetimes,counts,sell,level):
    res=[]
    for i in range(n):
        res.append(run(practicetimes,counts,sell,level))
    re=np.array(res)
    ave=np.average(re)
    std=np.std(re)
    return ave,std


def run_s (practicetimes, days):
    band=Band(0,1,0)#直接在这里修改
    promote=0
    disband=0
    concertcount=0
    for i in range(days):
        if promote==1:
            band.l+=1
            promote=0
            continue
        if disband==1:
            break
        if concertcount==practicetimes:
            band.l-=0#还有这里
        if concertcount<practicetimes:
            band,disband,promote=concert(band,500)#和这里
        else:
            band,disband,promote=concert(band,1000)
        concertcount+=1
    return band.s,band.l

def trail_s(n,practicetimes,days):
    res_s=[]
    res_l=[]
    for i in range(n):
        s,l=run_s(practicetimes,days)
        res_s.append(s)
        res_l.append(l)
    df = pd.DataFrame({'level':res_l,
                  'sell':res_s,},
                 columns=['level','sell'])
    return [res_l,res_s]

def countLS(l_s):
    z=np.zeros([9,40],dtype = int)
    for i in range(len(l_s[0])):
        z[(l_s[0][i]-1)][int((l_s[1][i]/50))]+=1
    return z

plt.plot(s,l, '*',label='original values')
plt.plot(s_s, lvals, 'r',label='linearfit values')
plt.plot(s_s, lmars, 'b',label='bondary of sequence')
plt.xlabel('唱片销量')
plt.ylabel('乐队等级')
plt.legend(loc=4) # 指定legend在图中的位置，类似象限的位置
plt.title('分界线上方先500收益更高，下方先1000收益更高')
plt.show()

z_normed = normalize(z, axis=1, norm='max')
fig, ax = plt.subplots(1,1)
ax.imshow(z_normed,cmap=plt.cm.jet)
#z_normed = z / z.max(axis=1)
sns.regplot(x="sell",y="level",data=df)
path=r'C:\Users\zenqi\Desktop\马\yaogun\data.txt'
data=np.loadtxt(path, dtype=float, delimiter=',' )
x,y=np.split(data,indices_or_sections=(2,),axis=1) 
lr = LinearRegression()
lr.fit(x,y)
#clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovo')
#clf.fit(x, y.ravel())

    
    

