

```python
import xgboost as xgb
import scipy.stats as st
import sklearn.metrics as met
import pandas as pd
import time,datetime,random,os
import numpy as np
import matplotlib.pyplot as plt
path = r'K:\PSA2016投标项目_用户提供数据\trainSet'
```

### Part1: 
Deleting the columns with almost all identical or NA values
Deleting the columns with serious collinearity with any column before


```python
fileList = os.listdir(r'K:\PSA2016投标项目_用户提供数据\trainSet')
dataset = dict([(file.split(r'.')[0],pd.read_pickle(path+'\\'+file)) for file in fileList])

dataMulti = dataset
for k,v in dataMulti.items():
    dataMulti[k].alarm_id = dataset[k].alarm_id*int(k)
dataMulti = pd.concat([v for k,v in dataMulti.items()])
dataMulti = dataMulti.ix[dataMulti.alarm_id!=0,:]
```


```python
dataset67 = pd.read_pickle(path+'\\67.pkl')
dataMulti = dataMulti.loc[:,dataset67.columns]
dataMulti.to_pickle('data.pkl')
```


```python
def preprocess(dataMulti):
    for ii in range(1,dataMulti.shape[1]):
        try:
            dataMulti.ix[dataMulti.iloc[:,ii]=='Bad',ii] = float('nan')
        except TypeError:
            continue
    for ii in range(1,dataMulti.shape[1]):
        try:
            dataMulti.iloc[:,ii] = dataMulti.iloc[:,ii].astype('float')
        except ValueError:
            dataMulti.ix[np.logical_and(dataMulti.iloc[:,ii]!=0,dataMulti.iloc[:,ii]!=1),ii]=float('nan')
            dataMulti.iloc[:,ii] = dataMulti.iloc[:,ii].astype('float')
    
    return dataMulti
```


```python
dataMulti = preprocess(dataMulti)
```

```python

```


```python
# The first 42 columns are continuous variables, and the rest are categorical(binary variables!)
columns_Type = pd.DataFrame(np.concatenate((np.zeros((1,2)),np.ones((1,40)),np.zeros((1,dataset.columns.shape[0]-42))),axis = 1),columns=dataset.columns)

```


```python
def Clean_Data(dataset):
    freqNa = pd.DataFrame([dataset.iloc[:,ii].isnull().sum() for ii in range(dataset.shape[1])]).transpose()
    dataC = dataset.iloc[:,list((freqNa/dataset.shape[0]<1-0.01).iloc[0,:])]
    # remove collinearity, fillna() with mean for continuous variable, and mode for categorical variable
    dataC = dataC.iloc[:,0:2].join(dataC.iloc[:,2:].fillna(value = dict([(column,dataC.iloc[:,ii+2].mean()) for ii, column in enumerate(dataC.columns[2:42])])))
    dataC = dataC.iloc[:,:42].join(dataC.iloc[:,42:].fillna(value = dict([(column,st.mode(dataC.iloc[:,ii+42]).mode[0]) for ii, column in enumerate(dataC.columns[42:])])))
    dataC = dataC.drop(dataC.columns[2:][np.any(np.abs(np.tril(np.corrcoef(dataC.iloc[:,2:],rowvar = 0),-1))>0.9, axis = 1)],axis = 1)
    
    return dataC

def Standardize_Data(dataC, columns_Type):
    # standardize， to qunantile!!!
    dataZ  = dataC
    dataZ.iloc[:,np.any(columns_Type[dataC.columns].values,axis=0)] = ((dataC.iloc[:,np.any(columns_Type[dataC.columns].values,axis=0)].rank()-0.5)/dataC.shape[0]).apply(st.norm.ppf)
    return dataZ
    
```


```python
dataC = Clean_Data(dataMulti)
# standardize， to qunantile!!!
dataZ = Standardize_Data(dataC,columns_Type)
```

### part2:
#### Model Functions
Functions for models to train, validate and predict Y from data
#### Model Evaluation (functions)
Score function and predictor for evaluating machine learning models

### Types of Training Models (functions)
Different model types for training from sub training and validation sets, random seed and model parameters

Input:
* xt1, yt1: sub training sets for data
* xt2, yt2: sub validation sets for data
* seed: random seed for the model
* parmodel: parameters of the model

Processing:
1. Train the model
1. Print validation score and running time

Output:
* model: trained machine learning model


```python
def Score(y, yp, f = met.roc_auc_score):
    score = f(y, yp)
    print("Score: {:.4f}".format(score))
    return(score)
```


```python
xTrain = dataZ.iloc[:,2:]
yTrain = dataZ.iloc[:,1]
idx = np.arange(xTrain.shape[0])
random.shuffle(idx)
xTrain = xTrain.iloc[idx, :]
yTrain = yTrain.iloc[idx]
trainSet = xgb.DMatrix(xTrain,label = yTrain, missing = np.nan)
par = {'colsample_bylevel': 0.1, 'max_depth': 10, 'min_child_weight': 1, 'sub_sample': 1, 
           'eta': 0.1, "seed": 0, "objective": 'binary:logistic', 'eval_metric': 'logloss'}
model1 = xgb.train(params = par, dtrain = trainSet, num_boost_round = 35)

```


```python
xTrain1 = xTrain.iloc[:300,:]
xTest1 = xTrain.iloc[300:,:]
yTrain1 = yTrain.iloc[:300]
yTest1 = yTrain.iloc[300:]
trainSet = xgb.DMatrix(xTrain1,label = yTrain1, missing = np.nan)
testSet = xgb.DMatrix(xTest1,label = yTest1, missing = np.nan)
parval = [(trainSet,'train'), (testSet,'val')]
par = {'colsample_bylevel': 0.1, 'max_depth': 10, 'min_child_weight': 1, 'sub_sample': 1, 
           'eta': 0.1, "seed": 0, "objective": 'binary:logistic', 'eval_metric': 'logloss'}
model2 = xgb.train(params = par, dtrain = trainSet, num_boost_round = 35,evals = parval)
```

    [0]	train-logloss:0.657583	val-logloss:0.68398
    [1]	train-logloss:0.627958	val-logloss:0.668498
    [2]	train-logloss:0.596304	val-logloss:0.669744
    [3]	train-logloss:0.573863	val-logloss:0.664872
    [4]	train-logloss:0.554311	val-logloss:0.664346
    [5]	train-logloss:0.528591	val-logloss:0.666079
    [6]	train-logloss:0.510322	val-logloss:0.664526
    [7]	train-logloss:0.491876	val-logloss:0.660662
    [8]	train-logloss:0.476232	val-logloss:0.663263
    [9]	train-logloss:0.456988	val-logloss:0.665199
    [10]	train-logloss:0.443027	val-logloss:0.669124
    [11]	train-logloss:0.431041	val-logloss:0.666547
    [12]	train-logloss:0.419274	val-logloss:0.664311
    [13]	train-logloss:0.407289	val-logloss:0.66485
    [14]	train-logloss:0.397578	val-logloss:0.667534
    [15]	train-logloss:0.386836	val-logloss:0.66279
    [16]	train-logloss:0.375667	val-logloss:0.662005
    [17]	train-logloss:0.365894	val-logloss:0.667843
    [18]	train-logloss:0.356384	val-logloss:0.665951
    [19]	train-logloss:0.348358	val-logloss:0.672731
    [20]	train-logloss:0.340994	val-logloss:0.676374
    [21]	train-logloss:0.331629	val-logloss:0.677315
    [22]	train-logloss:0.325385	val-logloss:0.679921
    [23]	train-logloss:0.318705	val-logloss:0.682792
    [24]	train-logloss:0.312111	val-logloss:0.687419
    [25]	train-logloss:0.305047	val-logloss:0.690553
    [26]	train-logloss:0.297844	val-logloss:0.69292
    [27]	train-logloss:0.289775	val-logloss:0.696434
    [28]	train-logloss:0.28378	val-logloss:0.700652
    [29]	train-logloss:0.277022	val-logloss:0.701788
    [30]	train-logloss:0.270606	val-logloss:0.705934
    [31]	train-logloss:0.265767	val-logloss:0.708344
    [32]	train-logloss:0.261284	val-logloss:0.70976
    [33]	train-logloss:0.255384	val-logloss:0.711111
    [34]	train-logloss:0.25054	val-logloss:0.711941
    


```python
yTest1 = yTest1.reset_index(drop = True)
xTest1 = xTest1.reset_index(drop = True)
```


```python
plt.figure(1)
plt.plot(yTest1.sort_values().values)
plt.axis([0,100,0,1.1])
plt.plot(model1.predict(xgb.DMatrix(xTest1.iloc[yTest1.sort_values().index,:])))
plt.savefig('result.png')
```


```python
res = xgb.cv(params = par, dtrain = trainSet, num_boost_round = 10000,nfold=10, seed=0, stratified=True,  
             early_stopping_rounds=10, verbose_eval=1, show_stdv=True)  
```

    [0]	train-logloss:0.661082+0.00270476	test-logloss:0.673579+0.00765009
    [1]	train-logloss:0.633279+0.00283452	test-logloss:0.661164+0.0145542
    [2]	train-logloss:0.6077+0.00442854	test-logloss:0.650928+0.0168797
    [3]	train-logloss:0.584545+0.00490453	test-logloss:0.640547+0.0223365
    [4]	train-logloss:0.5645+0.00474038	test-logloss:0.635092+0.0242244
    [5]	train-logloss:0.544786+0.00550815	test-logloss:0.630108+0.0297674
    [6]	train-logloss:0.526734+0.00530377	test-logloss:0.625244+0.0340744
    [7]	train-logloss:0.511298+0.00509264	test-logloss:0.619189+0.038303
    [8]	train-logloss:0.494767+0.0050796	test-logloss:0.616154+0.0407345
    [9]	train-logloss:0.478064+0.00621021	test-logloss:0.61301+0.0440511
    [10]	train-logloss:0.463587+0.00662614	test-logloss:0.612198+0.0489207
    [11]	train-logloss:0.44975+0.0062365	test-logloss:0.609371+0.0525097
    [12]	train-logloss:0.436968+0.0063645	test-logloss:0.608056+0.054633
    [13]	train-logloss:0.424849+0.00666948	test-logloss:0.608166+0.0570401
    [14]	train-logloss:0.412507+0.00669003	test-logloss:0.605777+0.0573922
    [15]	train-logloss:0.402035+0.00710323	test-logloss:0.605038+0.0585696
    [16]	train-logloss:0.390533+0.00742393	test-logloss:0.602724+0.0618581
    [17]	train-logloss:0.380825+0.007735	test-logloss:0.60386+0.0619058
    [18]	train-logloss:0.371653+0.00771095	test-logloss:0.604377+0.0611104
    [19]	train-logloss:0.362164+0.00765598	test-logloss:0.604179+0.0630912
    [20]	train-logloss:0.353745+0.00810136	test-logloss:0.602823+0.0636909
    [21]	train-logloss:0.346013+0.00845277	test-logloss:0.602457+0.06597
    [22]	train-logloss:0.338831+0.00860633	test-logloss:0.602613+0.0659786
    [23]	train-logloss:0.331613+0.00869396	test-logloss:0.601107+0.0679412
    [24]	train-logloss:0.324433+0.00856168	test-logloss:0.60165+0.0697563
    [25]	train-logloss:0.318511+0.00862416	test-logloss:0.601066+0.0708793
    [26]	train-logloss:0.312016+0.00829313	test-logloss:0.603181+0.0719745
    [27]	train-logloss:0.305542+0.00828905	test-logloss:0.602862+0.0724839
    [28]	train-logloss:0.299511+0.00832156	test-logloss:0.604551+0.0743812
    [29]	train-logloss:0.293293+0.00836702	test-logloss:0.604892+0.0737296
    [30]	train-logloss:0.287615+0.00800583	test-logloss:0.607803+0.0731132
    [31]	train-logloss:0.282186+0.00770772	test-logloss:0.606763+0.0741115
    [32]	train-logloss:0.276537+0.0075884	test-logloss:0.608833+0.0745149
    [33]	train-logloss:0.271721+0.00758795	test-logloss:0.609732+0.0748102
    [34]	train-logloss:0.26687+0.00764257	test-logloss:0.611169+0.0745158
    


```python
met.log_loss(["spam", "ham", "ham", "spam"],[[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
```




    0.21616187468057912




```python

```
