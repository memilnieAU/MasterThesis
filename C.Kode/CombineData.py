"""
dette script er til at loade data fra en fil, filtrere og eksportere det igen
Dette skal køres for hver movement, der er ikke lavet højre og venstre side notering endnu

De loadede data skal lægge i TestMeasurement2 og de bliver gemt i "Test"
    
"""

#%% import copy
import importlib
import os

import Backbone
importlib.reload(Backbone)
from Backbone import Lejring,Movement, Section, Sensor, SensorData, Trigger

import DataHelper
importlib.reload(DataHelper)
from DataHelper import Dataloader, DataExtractor,DataFiltre

import matplotlib.pyplot as plt

import numpy as np



from sklearn import metrics
from sklearn.metrics import classification_report

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Backbone import Movement


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc 
from sklearn.preprocessing import label_binarize
from itertools import cycle

loader = Dataloader()

#Change the movement here
movement = Movement.slag

movement = Movement.fist
movement = Movement.piano
movement = Movement.grib


#%% Load data from all subject
"""
Denne section vil loade de tidligere generede signaler og sætte dem sammen til et samlet signal
Loader kun control data
"""

#load name
printThis = False
plotThis = False
folder = "Sub6"
RandomSubFolder = "RandomSub1"

folders = ["Sub8"]
folders = ["Sub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8","RandomSub1"]


for folder in folders:

    foldername= folder+"\\Test\\"
    extenstion_c = "_Control.csv"
    extenstion_i = "_Input.csv"
    extenstion_l = "_Label.csv"
    fileNames =    ["ExportData_NA_Back"]
    fileNames =    "ExportData_Piano_Back" ,"ExportData_Fist_Back", "ExportData_Grib_Back" , "ExportData_Slag_Back"
    if folder == RandomSubFolder:
        fileNames =    ["ExportData_NA_Back"]

    extenstions = [extenstion_c,extenstion_i,extenstion_l]

    for extenstion in extenstions:
        CombinedData= pd.DataFrame()
        for name in fileNames:
            d = pd.read_csv(foldername+name+extenstion)

            
            if len(CombinedData) == 0:
                CombinedData = d
            else:
                CombinedData = CombinedData._append(d,ignore_index=True)
                

        signal=CombinedData

        if printThis: 
            print("np.min(signal)")
            print(np.min(signal))
            print("np.max(signal)")
            print(np.max(signal))
        
        

        if extenstion != extenstion_l:
            signal_min = np.min(signal,axis=0).values#signal.min(dim=1, keepdim=True).values  # Minimum along the length axis
            signal_max = np.max(signal,axis=0).values#signal.max(dim=1, keepdim=True).values  # Maximum along the length axis
            normalized_signal = (signal - signal_min) / (signal_max - signal_min + 1e-8)  # Add epsilon to avoid division by zero
            
            if printThis: 
                print("np.min(normalized_signal)")

                print(np.min(normalized_signal))
                print(np.max(normalized_signal))
            CombinedData = normalized_signal
        
        
        if printThis: 
            print("CombinedData.shape")
            print(CombinedData.shape)
            CombinedData.info()

        d = CombinedData
        # Crate a new timeline
        timeline = np.linspace(0,len(d)/100,len(d))
        d['Timeline_s'] = timeline

        d = d.drop(['Timeline_s','Unnamed: 0'], axis='columns')
        
        if plotThis:
            plt.figure(figsize=(15,2))
            plt.title(name)
            megIndex = [i for i in d.columns if 'meg' in i]
            megIndex = [i for i in d.columns if '' in i]
            print(megIndex)
            plt.plot(timeline,(d[megIndex]),label=megIndex)

            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.show()

        CombinedData = d
        if extenstion == extenstion_l:
            CombinedData['Timeline_s'] = timeline


        newpath = foldername+folder+"Combined"+extenstion
        CombinedData.to_csv(newpath)

        if printThis: 
            CombinedData.info()
            display(CombinedData.head())
        
        print("\n Saving path")
        print(newpath)
    


#%% Combine all subject anc combine them in one variable "Data[Subjectname]"

folder = "Sub4"
folders = ["Sub6","Sub7"]
folders = ["RandomSub1","Sub4","Sub7"]
folders = ["Sub2","Sub4","Sub7"]
folders = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
folders = ["RandomSub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
folders = ["Sub8"]
folders = ["RandomSub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8"]

folders = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8","RandomSub1"]


Data = {}

printThis = False
for folder in folders:
    foldername= folder+"\\Test\\"
    extenstion_c = "_Control.csv"
    extenstion_i = "_Input.csv"
    extenstion_l = "_Label.csv"
    fileNames =    "ExportData_Piano_Back" ,"ExportData_Fist_Back", "ExportData_Grib_Back" , "ExportData_Slag_Back"
    if folder == RandomSubFolder:
        fileNames =    ["ExportData_NA_Back"]
    extenstions = [extenstion_c,extenstion_i,extenstion_l]


    c_signal = pd.read_csv(foldername+folder+"Combined"+extenstion_c)
    i_signal = pd.read_csv(foldername+folder+"Combined"+extenstion_i)
    l_signal = pd.read_csv(foldername+folder+"Combined"+extenstion_l)

    def GetInfo(signal):
        try:
            signal = signal.drop(['Unnamed: 0'], axis='columns')
        except:
            print("no unnamed row was found")
        valueinfo = pd.DataFrame()
        nanSignal = np.isnan(signal)
        NaNs = np.sum(nanSignal*1,axis=0)
        valueinfo['nan'] = NaNs
        valueinfo['sample'] = len(signal)
        valueinfo['min'] = np.min(signal,axis=0)
        valueinfo['max'] = np.max(signal,axis=0)
        valueinfo['mean'] = np.mean(signal,axis=0)
        valueinfo['std'] = np.std(signal,axis=0)
        return signal,valueinfo



    c_signal, c_valueinfo = GetInfo(c_signal)
    i_signal, i_valueinfo = GetInfo(i_signal)
    l_signal,l_valueinfo = GetInfo(l_signal)
    
    NaVector = l_signal["Movement"] == 0.0
    print(np.sum(NaVector))
    s = l_signal["Movement"]
    s[NaVector] = 1.0
    
    print(np.sum(s == 0.0))
    l_signal["Movement"] = s
    
    
    NaVector = np.isnan(c_signal)
    # print(np.sum(NaVector))
    s = c_signal
    c_signal = s
    NaVector = np.isnan(c_signal)
    # print(np.sum(NaVector))
    
    
    NaVector = np.isnan(c_signal)
    c_signal[NaVector] = 0
    NaVector = np.isnan(c_signal)
    
    
    NaVector = np.isnan(i_signal)
    i_signal[NaVector] = 0
    NaVector = np.isnan(i_signal)
        
    
    print("_____________")
    c_signal, c_valueinfo = GetInfo(c_signal)
    print(c_valueinfo)
    print("_____________")
    i_signal, i_valueinfo = GetInfo(i_signal)
    print(i_valueinfo)
    print("_____________")
    l_signal,l_valueinfo = GetInfo(l_signal)
    
        
    if printThis:    
        display(l_valueinfo)
        display(c_valueinfo)
        display(i_valueinfo)

    
    thisData = {}
    thisData['c_signal'] = c_signal
    thisData['i_signal'] = i_signal
    thisData['l_signal'] = l_signal
    thisData['c_valueinfo'] = c_valueinfo
    thisData['i_valueinfo'] = i_valueinfo
    thisData['l_valueinfo'] = l_valueinfo
    
    
    print(folder + " is processed")
    
    Data[folder]  = thisData
        # valueinfo = np.concatenate(mi,ma)
        
#%% Plot meta data from one subject


folders = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
folders = ["RandomSub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
folders = ["Sub8"]
folders = ["RandomSub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8"]

folders = ["Sub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8","RandomSub1"]


k = str(Data.keys())
print(k)
thisSub = "RandomSub1"

k = str(Data[thisSub].keys())
print(k)
print()
print(f"Show data form subject: {thisSub}")
print(Data[thisSub]["i_valueinfo"])
print()
print(Data[thisSub]["c_valueinfo"])
print()
print(Data[thisSub]["l_valueinfo"])

    
#%% Plot Movement distribution from each subject

movement_mapping = {
  0.0: "NA",
  1.0: "No movement",
  2.0: "piano",
  3.0: "fist",
  4.0: "grib",
  5.0: "punch"
}

thisSub = "Sub8"
Subs = ["RandomSub1","Sub4","Sub7"]
Subs = ["Sub2","Sub4","Sub7"]
Subs = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
Subs = ["RandomSub1","Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]
Subs = ["Sub8"]
Subs = ["RandomSub1"]#,"Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8"]

Subs = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8","RandomSub1"]


BigMeta_Probs = pd.DataFrame()
BigMeta_Counts = pd.DataFrame()
for thisSub in Subs:
    l_signal = Data[thisSub]["l_signal"]
    OneSegment_labled_samplewise = pd.Series(l_signal['Movement'])
    OneSegment_labled_samplewise_named = OneSegment_labled_samplewise.map(movement_mapping)
    # l_signal['Movement_named'] = OneSegment_labled_samplewise_named


    OneSegment_labled_samplewise_named.value_counts()

    #calculate the procentage
    InfoL0 = OneSegment_labled_samplewise_named.value_counts(dropna=False)
   
    InfoL1 = OneSegment_labled_samplewise_named.value_counts(dropna=False,normalize=True)
    

    info = pd.concat([InfoL0, np.round(InfoL1,2)], axis=1)
    print(f"Information om movement fordeling for {thisSub}")
    # BigMeta[thisSub] = info
    # display(info)
    # print(InfoL0.keys())
    BigMeta_Prob = pd.DataFrame({
                 "Sub"+thisSub[-1]: np.round(InfoL1,2)},
                 InfoL0.keys())
    
    BigMeta_Count = pd.DataFrame({"Sub"+thisSub[-1]: InfoL0},
                 InfoL0.keys())
    BigMeta_Probs = pd.concat([BigMeta_Probs, BigMeta_Prob], axis=1)
    BigMeta_Counts = pd.concat([BigMeta_Counts, BigMeta_Count], axis=1)
    
display(BigMeta_Probs)

BigMeta_Counts["Sum"] =  BigMeta_Counts.sum(axis=1)
BigMeta_Counts["Tot Len_s"] =  BigMeta_Counts["Sum"]/100
BigMeta_Counts["Len_min"] =  np.floor(BigMeta_Counts["Tot Len_s"]/60)
BigMeta_Counts["Len_sec"] =  BigMeta_Counts["Tot Len_s"]%60

display(BigMeta_Counts)


d = BigMeta_Counts["Tot Len_s"].sum(axis=0)
print(d)
d = BigMeta_Counts["Len_min"].sum(axis=0)
print(d)
d = BigMeta_Counts["Len_sec"].sum(axis=0)
print(d)
# display(BigMeta)
    



 


#%% Plot One subject Interactive

def plotInteractive(df):
    # fig = go.Figure()
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    print("colors " + str(len(colors)))
    print("df.columns " + str(len(df.columns)))
    
    # fig.add_trace(
    # go.Scatter(x=[2, 3, 4], y=[4, 5, 6], name="yaxis2 data"),

    # secondary_y=True,
        # )
    
    
    # set up multiple traces
    for col in df.columns:
        if any(element in col for element in ["y_pred","y_true"]):
            print("PLOT YPRED")
            fig.add_trace(go.Scatter(x=df.index,
                                    y=df[col],
                                    name  = col,
                                    visible=True
                                    ),
                                    secondary_y=True
                        )
        
        else:
            fig.add_trace(go.Scatter(x=df.index,
                                y=df[col],
                                name  = col,
                                visible=True
                                )
                        )
            

    um = [ {} for _ in range(len(df.columns)) ]
    buttons = []
    menuadjustment = 0.15
    menuadjustment = 0.17

    buttonX = -0.1
    buttonY = 2 + menuadjustment
    for i, col in enumerate(df.columns):
        button = dict(method='restyle',
                    label=col,
                    visible=True,
                    args=[{'visible':True,
                            'line.color' : colors[i%10]}, [i]],
                    args2 = [{'visible': False,
                                'line.color' : colors[i%10]}, [i]],
                    )

        # adjust some button features
        buttonY = buttonY-menuadjustment
        um[i]['buttons'] = [button]
        um[i]['showactive'] = False
        um[i]['y'] = buttonY
        um[i]['x'] = buttonX

    # add a button to toggle all traces on and off
    button2 = dict(method='restyle',
                label='All',
                visible=True,
                args=[{'visible':True}],
                args2 = [{'visible': False}],
                )
    # assign button2 to an updatemenu and make some adjustments
    um.append(dict())
    um[i+1]['buttons'] = [button2]
    um[i+1]['showactive'] = True
    um[i+1]['y']=buttonY - menuadjustment
    um[i+1]['x'] = buttonX
        
    # add dropdown menus to the figure
    fig.update_layout(showlegend=True, updatemenus=um)

    fig.update_xaxes(
        rangeslider_visible=True
    )
    # adjust button type
    for m in fig.layout.updatemenus:
        m['type'] = 'buttons'

    # f = fig.full_figure_for_development(warn=False)
    fig.show()
       
thisSub = "Sub8"

thisSub = "RandomSub1"
    
c_signal = Data[thisSub]["c_signal"].copy()
i_signal = Data[thisSub]["i_signal"].copy()
l_signal = Data[thisSub]["l_signal"].copy()
print(str(c_signal.shape) + str(c_signal.keys()))
print(str(i_signal.shape)+ str(i_signal.keys()))
print(str(l_signal.shape) + str(l_signal.keys()))

plotdata = i_signal
plotdata['y_true'] = l_signal['Movement']
plotdata['Timeline_s'] = l_signal['Timeline_s']
plotdata = plotdata.set_index(['Timeline_s'])
print(str(plotdata.shape) + str(plotdata.keys()))


plotInteractive(plotdata)

#%% 

c = False
if c:
    signal = c_signal.copy()
    signalMeta = l_signal.copy()
    cText = " Arm Dataset"
    
    

    Data2Save = signal
    Lable2Save = signalMeta.drop(columns= ["Lejring","Timeline_s"])
    print(len(Lable2Save))

    display(Data2Save.head())

    print(len(Data2Save))



    with open('Data2Save_Arm_random.npy', 'wb') as f:
        np.save(f, Data2Save)
        np.save(f, Lable2Save)
else: 
    signal = i_signal.copy()
    signalMeta = l_signal.copy()
    cText = "Head Dataset"
    
    Data2Save = signal
    Data2Save = Data2Save.drop(columns= ["y_true","Timeline_s"])
    
    Lable2Save = signalMeta.drop(columns= ["Lejring","Timeline_s"])
    print(len(Lable2Save))

    display(Data2Save.head())

    print(len(Data2Save))



    with open('Data2Save_Head_random.npy', 'wb') as f:
        np.save(f, Data2Save)
        np.save(f, Lable2Save)
    
    


#%%
with open('Data2Save_Arm_random.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
print(a.shape)
print(b.shape)

with open('Data2Save_Head_random.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
print(a.shape)
print(b.shape)



#%% Fit reg to a partial of a signal


def MakeConfusionMatrix(model,X,y,target_names=["NO","Piano","Fist","Grib","Punch"],thisSub="", ModelName = "LogisticRegression",printReport = False):
    classes = target_names
    pred = model.predict(X)
    cf_matrix = confusion_matrix(y, pred)

    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    # print(cf_matrix)
    df_cm_norm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])

    fig = plt.figure(figsize = (16,5))
    plt.subplot(1,2,1)
    ax = sns.heatmap(df_cm, annot=True,fmt="2g")
    ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    plt.title(f'Confusion matrix', y=1.04)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.subplot(1,2,2)
    ax = sns.heatmap(df_cm_norm, annot=True,fmt=".2g")
    ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    plt.title(f'Confusion matrix in %', y=1.04)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    
    fig.suptitle(f'Confusion matrix for {ModelName} on {thisSub}', y=1.02, fontsize=16,) 
    plt.show()
    
    
    if printReport:
        print(classification_report(y, pred,target_names=classes,labels=np.unique(y) ,zero_division=0 ))
        print(classification_report(y, pred,labels=np.unique(pred) ,zero_division=0 ))



thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

signal = c_signal
#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


# instantiate the model (using the default parameters)
logreg_sub7_Parcial = LogisticRegression(random_state=16)

# fit the model with data
logreg_sub7_Parcial.fit(X_train, y_train)



target_names = list(movement_mapping.values())[1:]

MakeConfusionMatrix(logreg_sub7_Parcial,X_train,y_train,thisSub=thisSub+" - Train dataset",ModelName="logreg_sub7_Parcial")
MakeConfusionMatrix(logreg_sub7_Parcial,X,y,thisSub=thisSub+" - Full Dataset",ModelName="logreg_sub7_Parcial")
MakeConfusionMatrix(logreg_sub7_Parcial,X_test,y_test,thisSub=thisSub+" - Test Dataset",ModelName="logreg_sub7_Parcial")

# import the metrics clas
#%% Fit reg to one hole the signal


thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

c = False
cText = ""
if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"



#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

logreg_sub7 = LogisticRegression(random_state=16)

logreg_sub7.fit(X, y)

MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7",thisSub=thisSub + cText)


thisSub = "Sub4"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
# print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable


MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7",thisSub=thisSub+ cText)



thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
# print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable


MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7",thisSub=thisSub+ cText)


#%% plot_roc_curve(y_test, y_pred)


def plot_roc_curve(y_test, y_pred,thisSub="", ModelName = "LogisticRegression",printReport = False):
    n_classes = len(np.unique(y_test))
    # print(y_test[0:5])
    # print(y_pred[0:5])
    y_test = label_binarize(y_test, classes=np.arange(n_classes))
    # y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_pred[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    #plt.figure(figsize=(10,5))
    fig = plt.figure(dpi=200)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink", linestyle=":", linewidth=4,)

    plt.plot(fpr["macro"], tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy", linestyle=":", linewidth=4,)

    colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color in zip(range(n_classes), colors):
        
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (ROC) curve")
        plt.legend()
    fig.suptitle(f'ROC curve for {ModelName} on {thisSub}', y=1.02, fontsize=16,) 
        


thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"



#split dataset in features and target variable
feature_cols = signal.keys()
# print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable




y_pred_proba = logreg_sub7.predict_proba(X)
plot_roc_curve(y, y_pred_proba,ModelName="logreg_sub7",thisSub=thisSub+ cText)


#%% plot proba[:,4] = proba[:,4]


thisSub = "RandomSub1"
thisSub = "Sub4"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

pred = logreg_sub7.predict(X)
proba = logreg_sub7.predict_proba(X)
plt.figure(figsize=(16,5))
plt.plot(y,color='b')
# plt.plot(i_signal)
proba[:,0] = proba[:,0]
proba[:,1] = proba[:,1]
proba[:,2] = proba[:,2]
proba[:,3] = proba[:,3]
proba[:,4] = proba[:,4]
plt.plot(proba)


# plt.plot(pred-0.2,color='r')
plt.show()




#%% Try to fit a Sub7 Reg, and then a other one based on the probs


thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

c = False
cText = ""
if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"



#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

logreg_sub7 = LogisticRegression(random_state=16)

logreg_sub7.fit(X, y)


pred = logreg_sub7.predict(X)
proba = logreg_sub7.predict_proba(X)



logreg_last7 = LogisticRegression(random_state=16)
logreg_last7.fit(proba, y)

pred = logreg_last7.predict(proba)
MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7 + Normal",thisSub=thisSub+ cText)
MakeConfusionMatrix(logreg_last7,proba,y,ModelName="logreg_last7 + Predict on last probs",thisSub=thisSub)






thisSub = "Sub4"
thisSub = "RandomSub1"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

pred = logreg_sub7.predict(X)
proba = logreg_sub7.predict_proba(X)
MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7 + Normal",thisSub=thisSub+ cText)
MakeConfusionMatrix(logreg_last7,proba,y,ModelName="logreg_last7 + Predict on last probs",thisSub=thisSub+ cText)

pred = logreg_last7.predict(proba)


plt.figure(figsize=(16,5))
plt.plot(pred-0.2,color='r')
plt.plot(l_signal["Movement"])
plt.show()


#%% Predict another Sub / RandomSub on the new combined model


thisSub = "RandomSub1"
thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

pred = logreg_sub7.predict(X)
proba = logreg_sub7.predict_proba(X)

MakeConfusionMatrix(logreg_sub7,X,y,ModelName="logreg_sub7",thisSub=thisSub)
MakeConfusionMatrix(logreg_last7,proba,y,ModelName="logreg_last",thisSub=thisSub)
pred = logreg_last7.predict(proba)


plt.figure(figsize=(16,5))
plt.plot(pred-0.2,color='r')
plt.plot(l_signal["Movement"])
plt.show()

#%% View meta data


thisSub = "Sub8"
c_signals = Data[thisSub]["c_signal"]
i_signals = Data[thisSub]["i_signal"]
l_signals = Data[thisSub]["l_signal"]
l_signals.info()

def GetInfo(signal):
    # if 
    # signal = signal.drop(['Unnamed: 0'], axis='columns')
    valueinfo = pd.DataFrame()
    nanSignal = np.isnan(signal)
    NaNs = np.sum(nanSignal*1,axis=0)
    valueinfo['nan'] = NaNs
    valueinfo['sample'] = len(signal)
    valueinfo['min'] = np.min(signal,axis=0)
    valueinfo['max'] = np.max(signal,axis=0)
    valueinfo['mean'] = np.mean(signal,axis=0)
    valueinfo['std'] = np.std(signal,axis=0)
    return signal,valueinfo

signal,valueinfo = GetInfo(i_signals)

print(valueinfo)

#%% Combine all subs and view metadata
Subs = ["Sub4"]
Subs = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7","Sub8"]

c_signals = pd.DataFrame()
i_signals = pd.DataFrame()
l_signals = pd.DataFrame()
l_signals.info()


for thisSub in Subs:
    raw = Data[thisSub]["c_signal"]
    c_signals = pd.concat([c_signals, raw])
    
    raw = Data[thisSub]["i_signal"]
    i_signals = pd.concat((i_signals, raw))
    
    raw = Data[thisSub]["l_signal"]
    l_signals = pd.concat((l_signals, raw))
    
print(len(c_signals))
print(len(i_signals))
print(len(l_signals))

l_signals.info()


signal,valueinfo = GetInfo(l_signals)
print(valueinfo)
signal,valueinfo = GetInfo(i_signals)
print(valueinfo)

#%% Fit Logreg to all subjects

if c:
    signal = c_signals.copy()
    signalMeta = l_signals.copy()
    cText = " Arm Dataset"
else: 
    signal = i_signals.copy()
    signalMeta = l_signals.copy()
    cText = " Head Dataset" 
    
#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = signalMeta["Movement"] # Target variable

logreg_allsub = LogisticRegression(random_state=16)

logreg_allsub.fit(X, y)


pred = logreg_allsub.predict(X)
proba = logreg_allsub.predict_proba(X)


MakeConfusionMatrix(logreg_allsub,X,y,ModelName="logreg_allsub + Normal",thisSub="All data"+ cText)


logreg_allsub_probs = LogisticRegression(random_state=16)

logreg_allsub_probs.fit(proba, y)
MakeConfusionMatrix(logreg_allsub_probs,proba,y,ModelName="logreg_allsub_probs + Normal",thisSub="All data"+ cText)


#%% Try all subs on Sub7 and the unknown RandomSub



thisSub = "Sub7"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

proba = logreg_allsub.predict_proba(X)

MakeConfusionMatrix(logreg_allsub,X,y,ModelName="logreg_allsub + Normal",thisSub="All data"+ cText)
MakeConfusionMatrix(logreg_allsub_probs,proba,y,ModelName="logreg_allsub_probs + Normal",thisSub="All data"+ cText)




thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable


proba = logreg_allsub.predict_proba(X)
MakeConfusionMatrix(logreg_allsub,X,y,ModelName="logreg_allsub + Normal",thisSub="All data"+ cText + " ON _ " + thisSub)
MakeConfusionMatrix(logreg_allsub_probs,proba,y,ModelName="logreg_allsub_probs + Normal",thisSub="All data"+ cText+ " ON _ " + thisSub)



thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"


#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

pred = logreg_allsub.predict(X)
proba = logreg_allsub.predict_proba(X)
pred_prob = logreg_allsub_probs.predict(proba)


plt.figure(figsize=(16,5))
plt.plot(y,color='b')
plt.plot(pred-0.1,color='r',alpha=0.5)
plt.plot(pred_prob-0.2,color='y',alpha=0.5)

# plt.plot(i_signal)
proba[:,0] = proba[:,0]
proba[:,1] = proba[:,1]
proba[:,2] = proba[:,2]
proba[:,3] = proba[:,3]
proba[:,4] = proba[:,4]
plt.plot(proba)
plt.legend(["Y_true","Y_Pred","Y_Pred_prob","No","Piano","Fist","Grib","Punch"])

# plt.plot(pred-0.2,color='r')
plt.show()








#%% Experiment with other LogReg parameteres

if c:
    signal = c_signals.copy()
    signalMeta = l_signals.copy()
    cText = " Arm Dataset"
else: 
    signal = i_signals.copy()
    signalMeta = l_signals.copy()
    cText = " Head Dataset" 
    
    

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)

X = signal[feature_cols] # Features
y = signalMeta["Movement"] # Target variable




model = LogisticRegression(solver='newton-cg',class_weight='balanced', C=0.01, multi_class='ovr',
                           random_state=0,n_jobs=-1)
model.fit(X, y)


# LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
#                    solver='liblinear', tol=0.0001, verbose=0, warm_start=False)



ModelScore =  model.score(X, y)

thisSub= "All data"
sol = model.get_params()["solver"]
MakeConfusionMatrix(model,X,y,ModelName="Model: " + sol,thisSub=thisSub +" " +str(ModelScore))

thisSub = "RandomSub1"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

MakeConfusionMatrix(model,X,y,ModelName="Model: " + sol,thisSub=thisSub + cText)




#%%

thisSub = "RandomSub1"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]
if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

sns.set_theme()
 

MakeConfusionMatrix(model,X,y,ModelName="liblinear",thisSub=thisSub)



 
thisSub = "Sub8"
c_signal = Data[thisSub]["c_signal"]
i_signal = Data[thisSub]["i_signal"]
l_signal = Data[thisSub]["l_signal"]

if c:
    signal = c_signal
    cText = " Arm Dataset"
else: 
    signal = i_signal
    cText = " Head Dataset"

#split dataset in features and target variable
feature_cols = signal.keys()
print(feature_cols)
X = signal[feature_cols] # Features
y = l_signal["Movement"] # Target variable

pred = logreg_allsub.predict(X)
proba = logreg_allsub.predict_proba(X)
plt.figure(figsize=(16,5))

timeline = np.linspace(0,len(X)/100,len(X))

plt.plot(timeline,X['meg:Left Head'],label='meg:Left Head')
plt.plot(timeline,X['meg:Right Head'],label='meg:Right Head')

plt.plot(timeline,pred, alpha=1.0,label="Pred Y",linewidth=2)
plt.plot(timeline,y,    alpha=1.0,label="True Y",linewidth=2)
proba[:,0] = proba[:,0]
proba[:,1] = proba[:,1]
proba[:,2] = proba[:,2]
proba[:,3] = proba[:,3]
proba[:,4] = proba[:,4]

x = range(len(pred))
proba = proba*5
plt.stackplot(timeline,proba[:,0],proba[:,1],proba[:,2],proba[:,3],proba[:,4],labels=["No","Piano","Fist","Grib","Punch"],alpha=0.5)

plt.legend()
plt.show()


#%%



import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import pandas as pd

def plotDataWithInteractive(data):

  fig = px.line(data, x='Timeline_s', y=['meg:Left Head','y_true','y_pred'], title='Time Series with Range Slider and Selectors')

  fig.update_xaxes(
      rangeslider_visible=True
      # ,
      # rangeselector=dict(
      #     buttons=list([
      #         dict(count=1, label="1m", step="month", stepmode="backward"),
      #         dict(count=6, label="6m", step="month", stepmode="backward"),
      #         dict(count=1, label="YTD", step="year", stepmode="todate"),
      #         dict(count=1, label="1y", step="year", stepmode="backward"),
      #         dict(step="all")
      #     ])
      # )
  )
  fig.show()


plotdata = X
plotdata['Timeline_s'] = timeline
plotdata['y_pred'] = pred
plotdata['y_true'] = y
plotDataWithInteractive(plotdata)

#%%
import datetime




def plotInteractive(df):
    fig = go.Figure()
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = px.colors.qualitative.Plotly
    print("colors " + str(len(colors)))
    print("df.columns " + str(len(df.columns)))
    
    # fig.add_trace(
    # go.Scatter(x=[2, 3, 4], y=[4, 5, 6], name="yaxis2 data"),

    # secondary_y=True,
        # )
    
    
    # set up multiple traces
    for col in df.columns:
        if any(element in col for element in ["y_pred","y_true"]):
            print("PLOT YPRED")
            fig.add_trace(go.Scatter(x=df.index,
                                    y=df[col],
                                    name  = col,
                                    visible=True
                                    ),
                                    secondary_y=True
                        )
        
        else:
            fig.add_trace(go.Scatter(x=df.index,
                                y=df[col],
                                name  = col,
                                visible=True
                                )
                        )
            

    um = [ {} for _ in range(len(df.columns)) ]
    buttons = []
    menuadjustment = 0.15

    buttonX = -0.1
    buttonY = 1 + menuadjustment
    for i, col in enumerate(df.columns):
        button = dict(method='restyle',
                    label=col,
                    visible=True,
                    args=[{'visible':True,
                            'line.color' : colors[i%10]}, [i]],
                    args2 = [{'visible': False,
                                'line.color' : colors[i%10]}, [i]],
                    )

        # adjust some button features
        buttonY = buttonY-menuadjustment
        um[i]['buttons'] = [button]
        um[i]['showactive'] = False
        um[i]['y'] = buttonY
        um[i]['x'] = buttonX

    # add a button to toggle all traces on and off
    button2 = dict(method='restyle',
                label='All',
                visible=True,
                args=[{'visible':True}],
                args2 = [{'visible': False}],
                )
    # assign button2 to an updatemenu and make some adjustments
    um.append(dict())
    um[i+1]['buttons'] = [button2]
    um[i+1]['showactive'] = True
    um[i+1]['y']=buttonY - menuadjustment
    um[i+1]['x'] = buttonX
        
    # add dropdown menus to the figure
    fig.update_layout(showlegend=True, updatemenus=um)

    fig.update_xaxes(
        rangeslider_visible=True
    )
    # adjust button type
    for m in fig.layout.updatemenus:
        m['type'] = 'buttons'

    # f = fig.full_figure_for_development(warn=False)
    fig.show()

plotdata['Timeline_s'] = timeline
plotdata = plotdata.set_index(['Timeline_s'])
plotdata.info()
# df.drop('x:Right Head')
# df.info()

# df.columsnames()
# df = df.cumsum()
plotInteractive(plotdata)
#%% Slut på Reg model



#Count segtions and duration of class:




y_True = plotdata['y_true']
y_Pred = plotdata['y_pred']

display(y_True.shape)
display(y_Pred.shape)

print(y_Pred.values[0:100])

newsig = y_Pred.values.copy()
for i in range(0,len(y_Pred.values)):
    if newsig[i] == 5:
            
        if newsig[i] == newsig[i-2]:
            newsig[i-1] = newsig[i]

        x = 60
        if i>x:
            if newsig[i] == newsig[i-x]:
                newsig[i-x:i-1] = newsig[i]


print(newsig[0:100])

plt.plot(y_Pred.values)
plt.plot(y_True.values+0.5)
plt.plot(newsig-0.5)
# plt.plot(y_PredDiff)
plt.show()


#%%



c = False
if c:
    signal = c_signals.copy()
    signalMeta = l_signals.copy()
    cText = " Arm Dataset"
else: 
    signal = i_signals.copy()
    signalMeta = l_signals.copy()
    cText = "Head Dataset"
    
    
display(signalMeta.info())
display(signal.info())


Data2Save = signal
Lable2Save = signalMeta.drop(columns= ["Lejring","Timeline_s"])
print(len(Lable2Save))

display(Data2Save.head())

print(len(Data2Save))



with open('Data2SaveSub18Head.npy', 'wb') as f:
    np.save(f, Data2Save)
    np.save(f, Lable2Save)




with open('Data2SaveSub18Head.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
print(a.shape)
print(b.shape)

#%%
with open('CombinedDataNa.npy', 'wb') as f:
    np.save(f, raw_all)
    np.save(f, lables_all)
    
    
with open('CombinedDataNa.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
    
print(a.shape)
print(b.shape)



# %%
import matplotlib.pyplot as plt

def p(t,Printhis=True):
    if Printhis:
        print(t)

def printInfo(a,addtext="",printthis = True):
    p(f"{addtext} max {np.max(a)}",Printhis = printthis)
    p(f"{addtext} min {np.min(a)}",Printhis = printthis)
    p(f"{addtext} shape {a.shape}",Printhis = printthis)
    p(f"{addtext} type {a.dtype}",Printhis = printthis)
    is_nan = np.isnan(a)
    p(f"{addtext} Have nan {is_nan.sum()}",Printhis = printthis)

def calcMeg(sig1,sig2,printThis=True):
        
    meg = (sig1**2+sig2**2)**0.5
    printInfo(meg,"meg",printThis)

    signal = meg
    signal_min = np.min(signal)#signal.min(dim=1, keepdim=True).values  # Minimum along the length axis
    signal_max = np.max(signal)#signal.max(dim=1, keepdim=True).values  # Maximum along the length axis
    normalized_signal = (signal - signal_min) / (signal_max - signal_min + 1e-8)  # Add epsilon to avoid division by zero
    printInfo(normalized_signal,"normalized_signal",printThis)
    return normalized_signal

    
    
def printRawNLable(raw,lables,showthis = False):
    if showthis:
            
        fig, ax1 = plt.subplots(figsize=(15,2))


        color1 = 'tab:red'
        color2 = 'tab:brown'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Amplitude', color=color1)
        ax1.plot(raw)#,label="signal2")

        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Movement', color=color)  # we already handled the x-label with ax1
        ax2.plot(lables, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 6)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped


        ax1.legend()
        plt.title("raw")

        plt.show()
def LoadExtractor_Raw_Lable(filepath,showPrint=False,showPlot=False):
    dinput = pd.read_csv(filepath)

    t = showPrint

    rh = dinput['meg:Right Hand']
    ra = dinput['meg:Right Arm']
    lh = dinput['meg:Left Hand']
    la = dinput['meg:Left Arm']

    t = False
    printInfo(rh,"rh",t)
    printInfo(ra,"ra",t)
    printInfo(lh,"lh",t)
    printInfo(la,"la",t)

    t = False
    rmeg = calcMeg(rh,ra,t)
    lmeg = calcMeg(lh,la,t)


    t = False
    printInfo(rmeg,"rmeg",t)
    print()
    printInfo(lmeg,"lmeg",t)

    raw = np.zeros((len(rmeg),2))
    t = False

    printInfo(raw,"raw",t)
    raw[:,0] = rmeg
    raw[:,1] = lmeg
    printInfo(raw,"raw",t)


    lables = dinput['Movement']
    printInfo(lables,"lables",t)



    printRawNLable(raw,lables,showPlot)
    return raw,lables




    
extenstion = "_Control.csv"
extenstion = "_Input.csv"
folder = "Sub4"
foldername= folder+"\\Test\\"

filepath = foldername+folder+"Combined"+extenstion
print(filepath)
raw,lables = LoadExtractor_Raw_Lable(filepath,showPlot=True)

printInfo(raw,"raw")
print()

is_nan = np.isnan(raw[:,1])
print(is_nan)
print(len(is_nan))
print(np.sum(is_nan))
print()

is_nan = np.isnan(raw[:,0])
print(is_nan)
print(len(is_nan))
print(np.sum(is_nan))

print()
printInfo(lables,"lables")
# printRawNLable(raw[:,1],is_nan*5,showthis=True)
raw[is_nan,1] = 4
printRawNLable(raw,lables,showthis=True)

#%%

#Read all files and make tensors to save them in seperated files:

folders = ["Sub2","Sub3","Sub4","Sub5","Sub6","Sub7"]


raw_all = np.zeros((0,2))   
lables_all = np.zeros((1,1))   
print(raw_all.shape)

lables_all = []
for folder in folders:
        

    extenstion = "_Control.csv"
    extenstion = "_Input.csv"
    print(folder) 
    foldername= folder+"\\Test\\"

    filepath = foldername+folder+"Combined"+extenstion
    print(filepath)
    raw,lables = LoadExtractor_Raw_Lable(filepath,showPlot=False)
    print(raw.shape)
    
    raw_all = np.concatenate((raw_all, raw))
    lables_all = np.concatenate((lables_all,lables))

#%%


print(lables_all.shape)
print(raw_all.shape)

printInfo(raw_all,"raw_all")
print()
printInfo(lables_all,"lables_all")



#%%

# 417799 sub 2,3,_,5,6,7
# 511199 sub 2,3,4,5,6,7
with open('CombinedDataNa.npy', 'wb') as f:
    np.save(f, raw_all)
    np.save(f, lables_all)
    
    
with open('CombinedDataNa.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
    
print(a.shape)
print(b.shape)
