
### This script will pre-process the dataset, prepare it for data-mining algorithm, and save them into database which is easier to reload
There are several serialization alternatives in pandas, including csv, hdf, msgpack, pickle. Here, though msgpack is an experimental library, I decide to use it. 


```python
import pandas as pd
import numpy as np
import os,time,re,math,sklearn,multiprocessing
import datetime as dt
import sklearn.preprocessing as prep
from matplotlib import pyplot as plt
from functools import partial
%matplotlib inline
path = r'K:\PSA2016投标项目_用户提供数据'
```

### This section  store all the user-defined function.


```python
def countHour(fileName):
    # This funciton will check if there are 24-hour dataset in file:fileName
        
    data = pd.read_csv(fileName).dropna()
    count = 1
    for ii in range(data.shape[0]-1):
        if data.iloc[ii,0]>data.iloc[ii+1,0]:
            count +=1
    return count           
```

### Part1: process the ALARM data
Since not every .csv file is of high quality, a selection is necessary to implement. 


```python
def pdt(str_dt):
    # convert datetime string(e.g. 2016-02-01 00:00:01.234) to a datetime
    return dt.datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S.%f')

os.chdir(path)
os.chdir(os.getcwd()+'\\01 CRANE ALARMS')
fileList = os.listdir()
# This dataFrame store all data
dataAlarm = pd.DataFrame()
for file in fileList:
    if file.find('.CSV')!=-1:
        try:
            if countHour(file)==1:
                dataAlarm2 = pd.read_csv(file, converters = {'timestamp':pdt,
                                                             'timestamp_utc':pdt,'generation_time_utc':pdt}).dropna()
                dataAlarm = dataAlarm.append(dataAlarm2,ignore_index = True)     
        except:
            continue
os.chdir(path)
freq = pd.value_counts(dataAlarm.alarm_id)
dataAlarm.alarm_id = dataAlarm.alarm_id.astype('category').cat.rename_categories(1+np.arange(len(freq))).astype('float')
```


```python
# dump data into a msg file
dataAlarm.to_msgpack('alarmDatabase.msg')
```


```python
# read data from this msg file
dataAlarm = pd.read_msgpack(r'K:\PSA2016投标项目_用户提供数据\alarmDatabase.msg')
```


```python
# Locked!  dataAlarm_indexed
#dataAlarm = pd.DataFrame([dataAlarm.iloc[ii,0].date() for ii in range(dataAlarm.shape[0])], index = range(dataAlarm.shape[0]),
    #                         columns=['date']).join(dataAlarm, how = 'left')
#dataAlarm = dataAlarm.set_index('date').truncate(before = dt.date(2016,2,1), after = dt.date(2016,6,30))
#dataAlarm_Feb = dataAlarm.set_index('date').truncate(before = dt.date(2016,2,2), after = dt.date(2016,2,29))
#dataAlarm_indexed = dataAlarm.set_index('date').reset_index().reset_index().set_index(['index','date'])
```


```python
dataAlarm = dataAlarm.reset_index().set_index('date').truncate(before = dt.date(2016,2,2), after = dt.date(2016,6,30))
```

####summarize Alarm data

因为alarm的数据记录的是alarm状态改变的数据，也就是说某个alarm出现第一次是产生，第二次是消失。


```python
# Locked! Plot the distribution of faultDuration for each fault(Top 20)
def faultDurationPlot(dataAlarm):
    faultTimeDuration = dataAlarm.iloc[:,1]-dataAlarm.iloc[:,13]
    dataAlarm = dataAlarm.join(pd.DataFrame(faultTimeDuration, columns = ['faultTimeDuration']),how = 'left')
    dataAbridged = dataAlarm.ix[dataAlarm.faultTimeDuration > dataAlarm.faultTimeDuration[1],:]
    freq = pd.value_counts(dataAbridged.alarm_id)
    
    # timeDuration, a list of lists which store the timeDurations of each fault 
    timeDuration = [dataAbridged.faultTimeDuration[dataAbridged.alarm_id == ii] for ii in freq.index]
    # Plot the top 20 fault
    plt.boxplot(timeDuration[:20], labels = freq.index[:20])
    return [[elem.max(), elem.min(),elem.mean()] for elem in timeDuration]
def fault_To_MultiBin(dataAlarm):
    # This function apply one-hot encoding to the fault
    OneHotEnc = prep.OneHotEncoder()
    OneHotEnc.fit(dataAlarm.alarm_id.reshape(-1,1))
    alarm_id_MultiBin = OneHotEnc.transform(dataAlarm.alarm_id.reshape(-1,1)).todense(0)
    return pd.DataFrame(alarm_id_MultiBin,columns = ['alarm{}'.format(ii+1) 
                                        for ii in range(alarm_id_MultiBin.shape[1])]).join(dataAlarm.timestamp,how = 'left')

def divideDataset(dataAlarm):
    #This function divide the dataset into two part: fault generating, and fault dissolved
    freq = pd.value_counts(dataAlarm.alarm_id) 
    return dataAlarm.ix[dataAlarm.timestamp_utc == dataAlarm.generation_time_utc,:],dataAlarm.ix[dataAlarm.timestamp_utc != dataAlarm.generation_time_utc,:]
```


```python
(da1,da2) = divideDataset(dataAlarm)
```

### Part2. Prep the data for model building


```python
def fileListIndex_Fun(dirPath,DIGIT):
    # User defined local variable: Month
    Month = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7}
    if DIGIT == 1:
        lag = 1
    else:
        lag = 0
        
    os.chdir(dirPath)
    subDirSet = os.listdir()

    fileListSet = []
    for subDir in subDirSet:
        os.chdir(os.getcwd()+'\\'+subDir)
        fileList = os.listdir()
        fileIndexList = []
        for fileName in fileList:
            # fileName 
            splitedStr = re.split('\.|\s',fileName)
            fileIndex = []
            fileIndex.append(fileName)
            # Only the startTime of each file is extracted, since the endTime is incorrectly recorded for some file.
            # Such as: 520 MH DIGITAL 01 FEB 2016 23 to 23.xls
            fileIndex.append(dt.datetime(int(splitedStr[4+lag]),Month[splitedStr[3+lag][:3].upper()],int(splitedStr[2+lag]),int(splitedStr[5+lag]),0,0))
            fileIndexList.append(fileIndex)
        fileIndexList = pd.DataFrame(fileIndexList,columns = ['fileName','startTime']).sort_values(by = 'startTime')
        
        fileListSet.append(fileIndexList)
        os.chdir(dirPath)
    # dirIndex is an embedded dict(dirIndex['FEB']['fileList'], dirIndex['FEB']['dir'])
    # A DataFrame is a great data structure for fileList!
    return dict([(elem.split()[1],{'dir':elem,'fileList':fileListSet[ii]}) for ii, elem in enumerate(subDirSet)])
def dataCombine(data1, data2):
    ''' This function combine two data dict into a new one'''
    if len(data1) == 0:
        return data2    
    for key,value in data2.items():
        data1[key] = data1[key].append(value)    
    return data1     

def readOrigFile(fileName):
    '''This function reads analog/digit excel file, and return a dict of data'''
    data = pd.read_excel(fileName).dropna(axis = 1, how = 'all')    
    return dict((data.columns[ii],data.iloc[1:,ii:ii+2].dropna()) for ii in range(1,data.shape[1],2))


def read_observation(dirIndex,timeStamp,path,DIGIT):
    ''' This is function read the historical analog data at timeStamp directly from .csv file
        This function need to be repeated for each observation (rows in the final data matrix)
    Question: How to search in an efficient way?    
    if digit file, then DIGIT = 1, else DIGIT = 0 
    '''
    Month = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7}
    Month = dict([(v,k) for (k,v) in Month.items()])
    if DIGIT:
        dirPath = path+'\\03 MH DIGITAL\\'+dirIndex[Month[timeStamp.month]]['dir'] # This is one useful result
    else:
        dirPath = path+'\\02 MH ANALOG\\'+dirIndex[Month[timeStamp.month]]['dir'] # This is one useful result
    fileList = dirIndex[Month[timeStamp.month]]['fileList']
    index_F = max(max(np.where(fileList.startTime<=timeStamp)))
    fileName = dirIndex[Month[timeStamp.month]]['fileList'].fileName.iloc[index_F] # This is another usefull result

    filePath = dirPath+'\\'+fileName
    dataAnalog = readOrigFile(filePath)
    # This section index the exact observation
    result = dict()
    for k,v in dataAnalog.items():
        tmp = v.iloc[:,0]<=timeStamp
        if len(max(np.where(tmp))):
            index = max(max(np.where(tmp)))
            result[k] = list(v.iloc[index,0:2])
        else:
            result[k] = [float('nan'),float('nan')]
             
    return result     
```

### selection a portion of data according to the frequency of each fault. For fault with high freq, assign a small proporation,
while for fault with lower freq, assign a relatively high ratio.

1. Since I will building a dedicated model for each alarm(This is a the One-VS-One strategy for mutliclass classification)
**The predication variable** used including: 8 analog variable, 46 digit variable, and 331 other alarms
2. Last, for each variable, except the current values, several historical values are also needed to
 take autocorrelation(Time series analysis) into considerration. 
 As a first attemp, the values at **1min, 5min, 20min and 1h ago **are adopted here. 



```python
def read_all_predicators_Analog(timestamp,dirIndex_Analog,path):
    # This funciton read all predictors for one case
      # analog
    da_Analog_Cur = read_observation(dirIndex= dirIndex_Analog, timeStamp = timestamp,path=path,DIGIT=0)
    da_Analog_1m = read_observation(dirIndex= dirIndex_Analog, timeStamp = timestamp-dt.timedelta(0,60,0),path=path,DIGIT=0)
    da_Analog_5m = read_observation(dirIndex= dirIndex_Analog, timeStamp = timestamp-dt.timedelta(0,300,0),path=path,DIGIT=0)
    da_Analog_20m = read_observation(dirIndex= dirIndex_Analog, timeStamp = timestamp-dt.timedelta(0,1200,0),path=path,DIGIT=0)
    da_Analog_1h = read_observation(dirIndex= dirIndex_Analog, timeStamp = timestamp-dt.timedelta(0,3600,0),path=path,DIGIT=0)
    da_Analog_Ele = np.array([[v[1] for k,v in da_Analog_Cur.items()],[v[1] for k,v in da_Analog_1m.items()],
                 [v[1] for k,v in da_Analog_5m.items()],[v[1] for k,v in da_Analog_20m.items()],
                 [v[1] for k,v in da_Analog_1h.items()]]).transpose().reshape(1,-1)
    return da_Analog_Ele
def read_all_predicators_Digit(timestamp,dirIndex_Digit,path):
    # This funciton read all predictors for one case
    # digit
    da_Digit_Cur = read_observation(dirIndex= dirIndex_Digit, timeStamp = timestamp,path=path,DIGIT=1)
    da_Digit_1m = read_observation(dirIndex= dirIndex_Digit, timeStamp = timestamp-dt.timedelta(0,60,0),path=path,DIGIT=1)
    da_Digit_5m = read_observation(dirIndex= dirIndex_Digit, timeStamp = timestamp-dt.timedelta(0,300,0),path=path,DIGIT=1)
    da_Digit_20m = read_observation(dirIndex= dirIndex_Digit, timeStamp = timestamp-dt.timedelta(0,1200,0),path=path,DIGIT=1)
    da_Digit_1h = read_observation(dirIndex= dirIndex_Digit, timeStamp = timestamp-dt.timedelta(0,3600,0),path=path,DIGIT=1)
    da_Digit_Ele = np.array([[v[1] for k,v in da_Digit_Cur.items()],[v[1] for k,v in da_Digit_1m.items()],
                            [v[1] for k,v in da_Digit_5m.items()],[v[1] for k,v in da_Digit_20m.items()],
                            [v[1] for k,v in da_Digit_1h.items()]]).transpose().reshape(1,-1)
    return da_Digit_Ele
    



def data_Prep(dataAlarm, alarm_id, path, N = 400):
    #Comments:  This function prepare the dataset for each alarm_id
    #Input:
    #
    #Output:
    # date     |Developer |action
    # 2017-1-6|Feng Fan   |Creat
    # The first timestamp when each variable has valid  record!!
    dirIndex_Analog = fileListIndex_Fun(path + r'\02 MH ANALOG',0)
    dirIndex_Digit = fileListIndex_Fun(path + r'\03 MH DIGITAL',1)    
    tmp = readOrigFile(path + r'\02 MH ANALOG\01 FEB'+'\\'+dirIndex_Analog['FEB']['fileList'].fileName.iat[0])
    timestamp1 = list(tmp.values())[0].iloc[0,0]
    tmp = readOrigFile(path + r'\03 MH DIGITAL\01 FEB'+'\\'+dirIndex_Digit['FEB']['fileList'].fileName.iat[0])
    timestamp2 = list(tmp.values())[1].iloc[0,0]
    timestamp3 = dataAlarm.timestamp.iloc[0]
    timestamp = max((timestamp1,timestamp2,timestamp3)) +dt.timedelta(0,3600,0)
    
    
    #1 : select N observation
    da1 = dataAlarm.iloc[np.logical_and(dataAlarm.alarm_id == alarm_id,dataAlarm.timestamp>=timestamp).values,:]
    [da1,tmp] = divideDataset(da1)
    if da1.shape[0]>=N/2:
        da1 =da1.iloc[np.random.choice(np.arange(da1.shape[0]),int(N/2),replace = False),:]
    da1.alarm_id = 1  
    
    da0 = dataAlarm.iloc[(dataAlarm.alarm_id != alarm_id).values,:]
    da0 = da0.iloc[np.random.choice(np.arange(da0.shape[0]),int(N/2),replace = False),:]
    da0.alarm_id = 0
    
    da = da1.append(da0)
    da = da[['timestamp','alarm_id']]
    #2： prepare the predictors 
    
    da_Analog = da_Digit = []
    
    print('start prepare data',time.strftime('%H:%M:%S',time.localtime()))
    
    # Make the pool of workers
    pool = multiprocessing.Pool(2) # The number of workers should be set according to the circumstance 
    
    partial_func_An = partial(read_all_predicators_Analog, dirIndex_Analog = dirIndex_Analog,path = path)
    partial_func_Di = partial(read_all_predicators_Digit, dirIndex_Digit = dirIndex_Digit,path = path)
    da_Analog = pool.map(partial_func_An, da.timestamp.tolist())
    print('Analog data preparation: finished')
    da_Digit = pool.map(partial_func_Di, da.timestamp.tolist())
    print('Digit data preparation: finished')

    da_Analog_DF = pd.DataFrame(da_Analog, columns = ['{}_Cur'.format(Ele) for Ele in da_Analog_Cur.keys()]+
                                ['{}_1m'.format(Ele) for Ele in da_Analog_Cur.keys()]+['{}_5m'.format(Ele) for Ele in da_Analog_Cur.keys()]+
                                ['{}_20m'.format(Ele) for Ele in da_Analog_Cur.keys()]+['{}_1h'.format(Ele) for Ele in da_Analog_Cur.keys()])
    da_Digit_DF = pd.DataFrame(da_Digit, columns = ['{}_Cur'.format(ELe) for Ele in da_Digit_Cur.keys()]+
                              ['{}_1m'.format(ELe) for Ele in da_Digit_Cur.keys()]+['{}_5m'.format(ELe) for Ele in da_Digit_Cur.keys()]+
                              ['{}_20m'.format(ELe) for Ele in da_Digit_Cur.keys()]+['{}_1h'.format(ELe) for Ele in da_Digit_Cur.keys()])
            
    return da.join(da_Analog_DF,how = 'left').join(da_Digit_DF, how = 'left')
```


```python
x = [[1,2],[3,4],[5,6]]
pool = ThreadPool(4)
s = pool.map(sum,x)
pool.close()
pool.join()
```


```python
re = data_Prep(dataAlarm, 69,path,400)
```

   
