"""
dette script er til at loade data fra en fil, filtrere og eksportere det igen
Dette skal køres for hver movement, der er ikke lavet højre og venstre side notering endnu

De loadede data skal lægge i TestMeasurement2 og de bliver gemt i "Test"
    
"""

#%% import copy

import importlib
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



import Backbone
importlib.reload(Backbone)
from Backbone import Lejring,Movement, Section, Sensor, SensorData, Trigger

import DataHelper
importlib.reload(DataHelper)
from DataHelper import Dataloader, DataExtractor,DataFiltre

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


loader = Dataloader()

#Change the movement here
movement = Movement.NA
movement = Movement.fist
movement = Movement.piano 
movement = Movement.grib
movement = Movement.slag
folder = "Sub2"
folder = "Sub3"
folder = "Sub4"
folder = "Sub5"
folder = "Sub6"
folder = "Sub7"
folder = "Sub8"
folder = "RandomSub1"



printoutput = False
ThisisRandomSub = True
# movementType = "Gribe"
# for movement in [Movement.piano , Movement.fist,Movement.grib,Movement.slag]:
# for movement in [Movement.grib ]:
for movement in [Movement.NoMovement]:

    plotline = 73

    if ThisisRandomSub:
        filename = "sub_RandomTask__fs_100.0_acc_data_1970-01-01 00_05_50.csv"
        filepath = loader.getFileFromMovement(folder=folder,movement=movement,filename=filename)
    else:
        filepath = loader.getFileFromMovement(folder=folder,movement=movement)
    print(filepath)
    rawdata = loader.LoadDataFromFilePath(filepath)
    # print("len rawdata: ",len(rawdata))



    trigger : Trigger = loader.GetTrigger(rawdata)
    trigger.SetTimeline(loader.GetTimeline(rawdata))

    # inputData,controlData = loader.GetInputAndControlData(rawdata)
    headData,controlData,armData,handData = loader.GetInputAndControlData(rawdata)
    inputData = headData.copy()

    headDataRaw = handData.copy()


    # trigger.print()
    extractor = DataExtractor()


    filter = DataFiltre()
        
    orden = 2
    highcut =40
    lowcut = 1
    def filterdata(data):
        data = data.astype(float)
        for i in range(0,len(data.columns)):
            filtered_data = filter.HighPass_filter(data.iloc[:, i], highcut=lowcut, fs=100, order=orden)
            # filtered_data = filter.butter_bandpass_filter(data.iloc[:, i], lowcut=lowcut, highcut=highcut, fs=100, order=5)
            data.iloc[:, i] = filtered_data.astype(float)
        # data.iloc[:300, :] = 0
        return data
        
    filtertype = f"Båndpass filteret, cutoff={lowcut}-{highcut}hz, orden={orden}"
    filtertype = f"Highpass filteret, cutoff={lowcut}hz, orden={orden}"
            
        
    inputData = filterdata(inputData)
    controlData = filterdata(controlData)

    headData = filterdata(headData)
    
    armData = filterdata(armData)
    handData = filterdata(handData)
    headDataFiltered = handData.copy()


    def RMSData(data,printer = False):
            
        sensorsInData = int(len(data.columns)/3)
        if printer:
            print("sensorsInInputdata: ",sensorsInData)
        for i in range(0,sensorsInData*3,3):
            # print("i: ",i)
            name = data.columns[i]
            name = name.split(":")[1]
            newname = "meg:"+name
            data[newname] = np.sqrt((data.iloc[:,i+0]**2)+(data.iloc[:,i+1]**2)+(data.iloc[:,i+2]**2))
        return data


    inputData = RMSData(inputData)
    controlData = RMSData(controlData)

    headData = RMSData(headData)
    
    armData = RMSData(armData)
    handData = RMSData(handData)
    headDataRMS = handData.copy()
    dataType = "hånd"
    

    #devide trigger into sections of Movements
    # m = Movement()
    #if everything is what is should be, this should be the same as the number of movements
    #and the first movement should be 1sec, the next 4sec and the last 7sec
    #The movements are devided by 5sec pause

    #make new signal with same length as the input data
    movementSignal = np.zeros(len(trigger.signal))-1

    if ThisisRandomSub:
        #onto first movement is the signal just noice, hence NA
        movementSignal[0:trigger.Idx[0]] = Movement.NA.value
        movementSignal[trigger.Idx[-1]:] = Movement.NA.value
    else: 
        movementSignal[0:trigger.Idx[0]] = Movement.NoMovement.value
        movementSignal[trigger.Idx[-1]:] = Movement.NoMovement.value

    maskIndex = []
    #change between movement.piano and movement.NoMovement every 5 sec triggerStartIdx
    for i in range(0,len(trigger.Idx)-1):
        if i%2 == 0:
            if ThisisRandomSub:   
                movementSignal[trigger.Idx[i]:trigger.Idx[i+1]] = Movement.NoMovement.value 
                
            else:
                movementSignal[trigger.Idx[i]:trigger.Idx[i+1]] = movement.value #Movement.NoMovement.value #Byt om hvis det er manuel
        else:
            movementSignal[trigger.Idx[i]:trigger.Idx[i+1]] = Movement.NoMovement.value
            

    trigger.SetMovementSiganl(movementSignal)

    #if manuel lableing
    manuelLabling = ThisisRandomSub
    if manuelLabling:
        #Make a movement signal that gose
        import numpy as np

        # Initialize the mask array with 0.0 ("NA")
        mask = np.full(25400, 0.0)

        # Define the mapping for each movement type
        movement_mapping = {
            0.0: "NA",
            1.0: "NoMovement",
            2.0: "piano",
            3.0: "fist",
            4.0: "grib",
            5.0: "punch"
        }

        # Define movement intervals with provided indices and mappings
        movement_intervals = [
            (464, 1165, 2.0),   # trig,piv
            (1165, 1659, 1.0),  # NoMovement
            (1659, 2095, 2.0),  # trig,piv
            (2095, 2642, 1.0),  # Omrokering (mapped as "fist")
            (2642, 2789, 5.0),  # trig,ph
            (2789, 3227, 1.0),  # NoMovement
            (3227, 3497, 5.0),  # trig,ph
            (3497, 4306, 1.0),  # NoMovement
            (4306, 4630, 2.0),  # trig,pih
            (4630, 5236, 1.0),  # NoMovement
            (5236, 5508, 2.0),  # trig,piv
            (5508, 5829, 1.0),  # NoMovement
            (5829, 6574, 4.0),  # trig,gv
            (6574, 6709, 1.0),  # NoMovement
            (6709, 7673, 4.0),  # trig,gh
            (7673, 8684, 1.0),  # NoMovement
            (8684, 9110, 2.0),  # trig,pih
            (9110, 9335, 1.0),  # NoMovement
            (9335, 9959, 3.0),  # trig,fh
            (9959, 10232, 1.0), # NoMovement
            (10232, 10498, 5.0),# trig,pv
            (10498, 10957, 1.0),# Omrokering (mapped as "fist")
            (10957, 11267, 3.0),# trig,fh
            (11267, 11815, 1.0),# NoMovement
            (11815, 12410, 3.0),# trig,fv
            (12410, 12783, 1.0),# NoMovement
            (12783, 13287, 5.0),# trig,ph
            (13287, 13622, 1.0),# NoMovement
            (13622, 13856, 4.0),# trig,gh
            (13856, 15002, 1.0),# NoMovement
            
            # (15002, 15002, 4.0),# trig,gv
            
            
            (15002, 15719, 4.0),# NoMovement
            (15719, 16083, 1.0),# trig,gv
            (16083, 16709, 4.0),# NoMovement
            (16709, 17146, 1.0),# trig,piv
            (17146, 17442, 2.0),# NoMovement
            (17442, 17852, 1.0),# trig,pv
            (17852, 18198, 5.0),# NoMovement
            (18198, 18798, 1.0),# trig,pih
            (18798, 19128, 2.0),# NoMovement
            (19128, 19464, 1.0),# trig,gh
            (19464, 20031, 4.0),# NoMovement
            (20031, 21033, 1.0),# trig,ph
            (21033, 21331, 5.0),# NoMovement
            (21331, 21785, 1.0),# trig,gh
            (21785, 22198, 4.0),# NoMovement
            (22198, 22567, 1.0),# trig,gh
            (22567, 23203, 4.0),# NoMovement
            (23203, 23374, 1.0),# trig,pih
            (23374, 23760, 2.0),# NoMovement
            (23760, 24109, 1.0),# trig,fh
            (24109, 24474, 3.0),# NoMovement
            (24474, 24682, 1.0),# trig,ph
            (24682, 24956, 5.0),# NoMovement
            (24956, 25168, 1.0) # NoMovement
        ]

        # Apply the intervals to the mask array
        for start, end, movement_type in movement_intervals:
            mask[start:end] = movement_type

        # Convert mask to movement names if needed
        mask_names = [movement_mapping[m] for m in mask]

        # Check the output
        print(mask[:50])  # Print first 50 values to verify
        print(movementSignal[:50])  # Print first 50 values to verify
        movementSignal = mask


    lejring = np.zeros(len(trigger.signal))
    lejring = lejring*Lejring.BackSide.value
    trigger.SetLejring(lejring)
    trigger.SetMovementSiganl(movementSignal)
    # trigger["movement"] = signal
    #plot the signal 



    #FJern alt som ikke er movement
    #Get mask of d[d['Movement'] != Movement.NA.value]
    mask = trigger.movement == Movement.NA.value
    mask = (~mask)

    if printoutput:
        print(len(trigger.timeline_s))
    trigger.timeline_s = trigger.timeline_s[mask]
    
    if printoutput:
        print(len(trigger.timeline_s))
    trigger.signal = trigger.signal[mask]
    trigger.movement = trigger.movement[mask]
    trigger.lejring = trigger.lejring[mask]



    inputData = inputData[mask]
    controlData = controlData[mask]

    headData = headData[mask]
    armData = armData[mask]
    handData = handData[mask]



    def GetMegIndex(data,printer=False):
            
        signal2Plot = data.iloc[:,-1]
        signal2Plot_max = signal2Plot.max()
        signal2Plot_min = signal2Plot.min()


        megindex = data.columns[data.columns.str.contains('meg')] 
        if printer:
            print("megindex: ",megindex[0])
            print("len megindex: ",len(megindex))
            print("len trigger: ",len(trigger.Idx))
            print(trigger.Idx)
            print(trigger.timeline_s[trigger.Idx])

        #ØØØØØØØØØØØØØØØØØØØSKIFT HER - Section skifter
        #Find index of if columname contain "meg"
        if printer:
            print("megindex: ",megindex[0])
            print("len megindex: ",len(megindex))
        return data,megindex

    headData,megindex =GetMegIndex(headData)
    # print(megindex)
    armData,megindex =GetMegIndex(armData)
    # print(megindex)
    handData,megindex =GetMegIndex(handData)
    
    if printoutput:
        print(megindex)


    def plotData(data,trigger,pltidx=0,windowLen=20,printer=False):
        plt.figure(figsize=(15,2))
        #lenght of view
        section = pltidx+windowLen

        # print("pltLineAt_sample " + str(startIdx))

        startIdx= trigger.Idx[pltidx]
        slutIdx = trigger.Idx[section]
        start = trigger.timeline_s[startIdx]-2
        slut = trigger.timeline_s[slutIdx]+2
        # start = 50
        # slut = 80

        megindex = data.columns[data.columns.str.contains('x')] 
        megindex = data.columns[data.columns.str.contains('y')] 
        megindex = data.columns[data.columns.str.contains('z')] 
        megindex = data.columns[data.columns.str.contains('meg')] 

        for i in megindex:
            signal2Plot = data[i]
            if printer:
                print(signal2Plot.shape)
            max = signal2Plot.max()
            min = signal2Plot.min()
            if printer:
                print(max)
                print(min)
            
                
            plt.plot(trigger.timeline_s,(signal2Plot/max)*5,label=signal2Plot.name)
            # plt.plot(trigger.timeline_s,(signal2Plot),label=signal2Plot.name)

           
           
            
        plt.plot(trigger.timeline_s,-trigger.signal,label='Trigger Signal',linewidth=2)

        plt.plot(trigger.timeline_s,trigger.movement,label='Movement Signal',color='black',linewidth=2)
        #fill when trugger.movement is not 0

        m = np.zeros(len(trigger.movement))
        m[trigger.movement>1] = 5
        plt.fill_between(trigger.timeline_s,m,color="r",alpha=0.2)
        m[trigger.movement>1] = -1
        plt.fill_between(trigger.timeline_s,m,color="r",alpha=0.2)



        pltLineAt = trigger.Idx[pltidx]
        plotline_s = trigger.timeline_s[pltLineAt]
        plotline_s = 32
        if printer:
            print("pltLineAt_sample " + str(pltLineAt))
            print("plotline_s " + str(plotline_s))
        plt.axvline(x=plotline_s, color='g', linestyle='--',label='Start' + "idx: " + str(pltLineAt),linewidth=4)


        pltLineAt = trigger.Idx[pltidx+1]
        plotline_s = trigger.timeline_s[pltLineAt]




        plotline_s = 33
        if printer:
            print("Extra line at: "+ str(plotline_s))
            print(trigger.Idx)
        plt.axvline(x=plotline_s, color='r', linestyle='--',label='Slut'+ "idx: " + str(pltLineAt),linewidth=4)

        if printer:
            print("Extra line at: " +str(plotline_s))


        #hline
        for (val,name ) in enumerate(Movement):
            plt.axhline(y=val, color='k', linestyle='--',linewidth=0.5) #label=name ,
            t = (str(val) + " "+ name.name )
            plt.text(slut+0.2,val-0.3,(t),ha='left')


        plt.xlabel = "Time [s]"
        plt.legend(bbox_to_anchor=(1.1, 1.05))


        plt.ylim((-1.5,6))
        plt.xlim((start,slut))
        # plt.xlim((31.5,33.5))
        
        plt.minorticks_on()   
        plt.tick_params(axis='x',which='minor', length=5, color='r',width=2)
        plt.title(f"{megindex}")
        plt.show()
        

    fig = plt.figure(figsize=[16,4])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.8, wspace=0.2,hspace=0.3)
    # ax.set_title('Rå data', fontsize= 12, fontweight="bold")
    fig.suptitle(f"Visualisering af 6. forsøgspersons {dataType} data \n {filtertype}", fontsize= 16, fontweight="bold")

    ax = plt.subplot(111)
    # ax2 = fig.add_subplot(122)

    # plt.figure(figsize=(15,2))
    m = np.zeros(len(trigger.movement))
    m[trigger.movement>1] = 5
    ax.fill_between(trigger.timeline_s,m,color="r",alpha=0.2)
    m[trigger.movement>1] = -1
    ax.fill_between(trigger.timeline_s,m,color="r",alpha=0.2)


    # plt.ylabel('Latitude') 
    # plt.xlabel('Longitude')


    start = 50
    slut = 120
    ax.plot(trigger.timeline_s,-trigger.signal,label='Trigger Signal',linewidth=2)

    ax.plot(trigger.timeline_s,trigger.movement,label='Movement Signal',color='black',linewidth=2)
    #fill when trugger.movement is not 0
    xIndex = handData.columns 
        
    scaleFactor = 2
    # scaleFactor = int(np.max(headDataRMS))
    for i in xIndex:
        signal2Plot = handData[i]
        max = signal2Plot.max()
        min = signal2Plot.min()
                
        # plt.plot(trigger.timeline_s,(signal2Plot[458:(len(trigger.timeline_s)+458)]/max)*10,label=signal2Plot.name)
        ax.plot(trigger.timeline_s,(signal2Plot)*scaleFactor,label=signal2Plot.name)

            #hline
        for (val,name ) in enumerate(Movement):
            ax.axhline(y=val, color='k', linestyle='--',linewidth=0.5) #label=name ,
            t = (str(val) + " "+ name.name )
            ax.text(slut+0.2,val-0.3,(t),ha='left')


    ax.legend(loc='upper right',ncol=5)


    
    ax.set_ylim((-1.5,6))
    ax.set_xlim((start,slut))
    # plt.xlim((31.5,33.5))

    ax.minorticks_on()   
    ax.tick_params(axis='x',which='minor', length=5, color='r',width=2)


    ax.set_xlabel("Time [s]")
    ax.set_ylabel('Magnitude (g-force)')
    ax.set_yticks([-1,0, 1, 2,3,4,5], [-1/scaleFactor,0/scaleFactor,1/scaleFactor,2/scaleFactor,3/scaleFactor,4/scaleFactor,5/scaleFactor])

    plt.show()



# signal2Plot = headDataFiltered[i]
# if printer:
#     print(signal2Plot.shape)
# max = signal2Plot.max()
# min = signal2Plot.min()
# if printer:
#     print(max)
#     print(min)

# plt.plot(trigger.timeline_s,(signal2Plot/max)*5,label=signal2Plot.name)


# signal2Plot = headDataRMS[i]
# if printer:
#     print(signal2Plot.shape)
# max = signal2Plot.max()
# min = signal2Plot.min()
# if printer:
#     print(max)
#     print(min)

# plt.plot(trigger.timeline_s,(signal2Plot/max)*5,label=signal2Plot.name)





    printoutput = True

    if printoutput:
            
        windowLen= 10
        pltidx = 0
        plotData(headData,trigger,windowLen=windowLen,pltidx=pltidx)
        # plotData(armData,trigger,windowLen=windowLen,pltidx=pltidx)
        # plotData(handData,trigger,windowLen=windowLen,pltidx=pltidx)

    printoutput = False

    if printoutput:
            
        print()
        print(f"{movement}")
        print(f"{np.min(np.min(headData,axis=0))} headData {np.max(np.max(headData,axis=0))}")
        print(f"{np.min(np.min(armData,axis=0))} armData  {np.max(np.max(armData,axis=0))}")
        print(f"{np.min(np.min(handData,axis=0))} handData {np.max(np.max(handData,axis=0))}")
    # print(f"{np.min(np.min(inputData,axis=0))}  {np.max(np.max(inputData,axis=0))}")


    """# %% Export data individuelt ikke Normaliseret endnu
    """


    exportfoldername= "output"
    if not os.path.exists(folder+"\\"+exportfoldername):
        os.makedirs(folder+"\\"+exportfoldername)
    foldername= folder+"\\"+exportfoldername+"\\"


    ExportLable = pd.DataFrame()
    ExportLable['Movement'] = trigger.movement
    ExportLable['Lejring'] = trigger.lejring
    ExportLable['Timeline_s'] = trigger.timeline_s

    if printoutput:
            
        display(ExportLable.head())
        display(ExportLable.info())


    name = "ExportData_"+movement.name+"_Back_Label.csv"

    if printoutput:
        print("name: ",foldername+name)
    ExportLable.to_csv(foldername+name)



    def ExportData(data,name,printer=False):
            
        ExportData = data

        ExportData['Timeline_s'] = trigger.timeline_s
        if printer:
            display(ExportData.head())
            display(ExportData.info())

        name = "ExportData_"+movement.name+"_Back_"+name+".csv"
        print("name: ",foldername+name)
        ExportData.to_csv(foldername+name)
        
    ExportData(headData,"Head")
    ExportData(armData,"Arm")
    ExportData(handData,"Hand")



#%% Display interactive the exported data

def plotInteractive(df,timeline):
        
    
    df['Timeline_s'] = timeline
    df = df.set_index(['Timeline_s'])

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
        if any(element in col for element in ["y_pred","y_true","Movement Signal"]):
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


# plt.plot(trigger.timeline_s,trigger.movement,label='Movement Signal',color='black',linewidth=2)

t = trigger.timeline_s.copy()


data = headData.copy()*5
data['Movement Signal'] = trigger.movement
plotInteractive(data,t)


data = handData.copy()
data['Movement Signal'] = trigger.movement
plotInteractive(data,t)


data = armData.copy()
data['Movement Signal'] = trigger.movement
plotInteractive(data,t)


#%%
"""
Denne section vil loade de tidligere generede signaler og sætte dem sammen til et samlet signal
Loader kun control data
"""

#load name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Backbone import Movement

folder = "Sub6"
folder = "Sub2"
folder = "Sub3"
folder = "Sub4"
folder = "Sub5"
folder = "Sub7"
folder = "Sub8"
folder = "RandomSub1"

subjectName = "6. forsøgspersons"
subjectName = "tilfældig 9. forsøgspersons"
exportfoldername= "output"
ThisisRandomSub= True
foldername= folder+"\\"+exportfoldername+"\\"

extenstion = "_Hand.csv"
extenstion = "_Arm.csv"
extenstion = "_Head.csv"
if ThisisRandomSub:
    fileNames =    ["ExportData_NoMovement_Back"]
else:
    fileNames =    "ExportData_Piano_Back" ,"ExportData_Fist_Back", "ExportData_Grib_Back" , "ExportData_Slag_Back"


printoutput = True



printoutput = False
def LoadExportData_SaveAsCombined(extenstion,moevment):
        
    CombinedData= pd.DataFrame()
    for name in fileNames:
        path = foldername+name+extenstion
        print(path)
        d = pd.read_csv(path)

        
        if len(CombinedData) == 0:
            CombinedData = d  
        else:
            CombinedData = CombinedData._append(d,ignore_index=True)
                
    if printoutput:
        display(CombinedData.info())

  
    
    # plt.plot(timeline,(CombinedData[megIndex]),label=megIndex)


    if extenstion != "_Label.csv":
    # if True:

        megIndex1 = [i for i in CombinedData.columns if 'x' in i]
        megIndex2 = [i for i in CombinedData.columns if 'y' in i]
        megIndex3 = [i for i in CombinedData.columns if 'z' in i]
        
        megIndex = np.append(megIndex1,megIndex2)
        megIndex = np.append(megIndex,megIndex3)
        # print(len(CombinedData[megIndex]))

        # print(megIndex)
        signal=CombinedData[megIndex]

        # print()
        # print("Total np.min(signal)")
        # print(np.min(signal))
        # print(np.max(signal))

        signal_mean = np.mean(signal)
        # print("Total signal_mean")
         
        # if printoutput:
        #     print(signal_mean)
        #     print(len(CombinedData[megIndex[0]]))
            
        
        centered_signal = signal - signal_mean
        
        signal_min = np.min(centered_signal)
        signal_max = np.max(centered_signal)
        siganl_maxmin = int(np.max([signal_max,signal_min]))+1
        if printoutput:
            print(siganl_maxmin)
            
        signal_max = siganl_maxmin
        signal_min = -siganl_maxmin
        normalized_signal = (centered_signal - signal_min) / (signal_max - signal_min)  # Scale to [0, 1]
        normalized_signal = normalized_signal * 1.0  # Scale to [-1, 1]
        # print(len(normalized_signal))

        # print()
        # print("np.min(normalized_signal)")
        # print(np.min(normalized_signal))
        # print(np.max(normalized_signal))
        # print(len(normalized_signal))
        # print(np.max(normalized_signal))
        
        
        signal_mean = np.mean(normalized_signal)
        # print("Total signal_mean")
        # print(signal_mean)

        CombinedData[megIndex] = normalized_signal[megIndex]
        
        
        
        megIndex = [i for i in CombinedData.columns if 'meg' in i]
        
        # print(megIndex)
        signal=CombinedData[megIndex]

        # print()
        # print("Total np.min(signal)")
        # print(np.min(signal))
        # print(np.max(signal))

        normalized_signal= signal
        signal_min = np.min(signal)#signal.min(dim=1, keepdim=True).values  # Minimum along the length axis
        signal_max = np.max(signal)#signal.max(dim=1, keepdim=True).values  # Maximum along the length axis
        normalized_signal = (signal - signal_min) / (signal_max - signal_min + 1e-8)  # Add epsilon to avoid division by zero
        # # print()
        # print(signal_min)
        # print()
        # print("np.min(normalized_signal)")
        # print(np.min(normalized_signal))
        # print(np.max(normalized_signal))

        CombinedData[megIndex] = normalized_signal[megIndex]
        
        
          #creat a new continues timeline
    timeline = np.linspace(0,len(CombinedData)/100,len(CombinedData))
    CombinedData['Timeline_s'] = timeline

    # print(len(CombinedData['Timeline_s']))
    # display(CombinedData.info)
    # print("FAFW")

    if extenstion == "_Label.csv":
        print(CombinedData.columns)
        return CombinedData["Movement"]
    
        
        
    
    
    if extenstion=="_Label.csv":
        megIndex = [i for i in CombinedData.columns if 'Move' in i]
        print(CombinedData.columns)
    else:
        megIndex = [i for i in CombinedData.columns if 'x' in i]
        megIndex = [i for i in CombinedData.columns if 'z' in i]
        megIndex = [i for i in CombinedData.columns if 'y' in i]
        
        
        megIndex1 = [i for i in CombinedData.columns if 'x' in i]
        megIndex2 = [i for i in CombinedData.columns if 'y' in i]
        megIndex3 = [i for i in CombinedData.columns if 'z' in i]
        
        megmegIndex = [i for i in CombinedData.columns if 'meg' in i]
        megIndex = np.append(megIndex1,megIndex2)
        megIndex = np.append(megIndex,megIndex3)
        megIndex = np.append(megIndex,megmegIndex)
        
        # megIndex = [i for i in CombinedData.columns if 'meg' in i]

   
    # printoutput = True
    if True:
    # if extenstion!="_Label.csv":        
    
    # if extenstion!="_Label.csv":
        # plotData(CombinedData,trigger,dataType,filtertype,megIndex)
            #    data,trigger,dataType,filtertype,columns2Plot 
        # plt.figure(figsize=(15,2))

        # plt.title(name + "             " + extenstion)
        # plt.plot(timeline,(CombinedData[megIndex]),label=megIndex)
        # plt.legend(bbox_to_anchor=(1.1, 1.05))
        # plt.show()
        # print()
        # print(CombinedData.shape)
        
        
            
        fig = plt.figure(figsize=[16,4])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                        top=0.8, wspace=0.2,hspace=0.3)
        # ax.set_title('Rå data', fontsize= 12, fontweight="bold")
        fig.suptitle(f"Visualisering af {subjectName} {dataType} data ", fontsize= 16, fontweight="bold")

        ax = plt.subplot(111)
        
        
        m = np.zeros(len(moevment))
        m[moevment>1] = 5
        ax.fill_between(CombinedData['Timeline_s'] ,m,color="r",alpha=0.2)
        m[moevment>1] = -1
        ax.fill_between(CombinedData['Timeline_s'] ,m,color="r",alpha=0.2)


        # plt.ylabel('Latitude') 
        # plt.xlabel('Longitude')

        
        start = 0
        t = CombinedData['Timeline_s'].values
        slut = t[-1]
        slut = 50
        # ax.plot(trigger.timeline_s,-trigger.signal,label='Trigger Signal',linewidth=2)

        ax.plot(CombinedData['Timeline_s'],moevment,label='Movement Signal',color='black',linewidth=2)
        #fill when trugger.movement is not 0
       
        scaleFactor = 5
        # scaleFactor = int(np.max(headDataRMS))
        # plt.plot(trigger.timeline_s,(signal2Plot[458:(len(trigger.timeline_s)+458)]/max)*10,label=signal2Plot.name)
        for id in megIndex:
            # print(len(trigger.timeline_s))
            # print(len(CombinedData[id]))
            # print(len(trigger.timeline_s))
            ax.plot(CombinedData['Timeline_s'] ,(CombinedData[id])*scaleFactor,label=id)
            # print(len(CombinedData[id]))
                #hline

        for (val,name ) in enumerate(Movement):
            ax.axhline(y=val, color='k', linestyle='--',linewidth=0.5) #label=name ,
            t = (str(val) + " "+ name.name )
            ax.text(slut+0.2,val-0.3,(t),ha='left')


        ax.legend(loc='upper right',ncol=5)


        # ax.set_ylim((-1.5,6))
        ax.set_xlim((start,slut))
        # plt.xlim((31.5,33.5))

        ax.minorticks_on()   
        ax.tick_params(axis='x',which='minor', length=5, color='r',width=2)


        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Magnitude (g-force)')
        ax.set_yticks([-1,0, 1, 2,3,4,5], [-1/scaleFactor,0/scaleFactor,1/scaleFactor,2/scaleFactor,3/scaleFactor,4/scaleFactor,5/scaleFactor])

        plt.show()
    
    
    # CombinedData['Movement'] = d['Movement']

    newpath = foldername+folder+"_Combined"+extenstion

    print(newpath)
    CombinedData.to_csv(newpath)
    # CombinedData.info()



extenstion = "_Label.csv"
moevment = LoadExportData_SaveAsCombined(extenstion,None)

dataType = "Øre"
extenstion = "_Head.csv"
LoadExportData_SaveAsCombined(extenstion,moevment)

dataType = "Arm"
extenstion = "_Arm.csv"
LoadExportData_SaveAsCombined(extenstion,moevment)

dataType = "Hånd"
extenstion = "_Hand.csv"
LoadExportData_SaveAsCombined(extenstion,moevment)







# %% Check for nan, der er fx nan i sub4
import matplotlib.pyplot as plt

def p(t,Printhis=True):
    if Printhis:
        print(t)

def printInfo(a,addtext="",printthis = True):
    # p(f"{addtext} max {np.max(a)}",Printhis = printthis)
    # p(f"{addtext} min {np.min(a)}",Printhis = printthis)
    # p(f"{addtext} shape {a.shape}",Printhis = printthis)
    # p(f"{addtext} type {a.dtype}",Printhis = printthis)
    is_nan = np.isnan(a)
    p(f"{addtext} \tHave nan {is_nan.sum()}",Printhis = printthis)
    return is_nan.sum()

folder = "RandomSub1"
folder = "Sub8"

exportfoldername= "output"

foldername= folder+"\\"+exportfoldername+"\\"
    

extenstion = "_Label.csv"
extenstion = "_Arm.csv"
extenstion = "_Head.csv"
extenstion = "_Hand.csv"

nans = 0
for extension in ["_Head.csv","_Hand.csv","_Arm.csv","_Label.csv"]:

    filepath = foldername+folder+"_Combined"+extenstion
    print(filepath)
    CombinedData = pd.read_csv(filepath)
    CombinedData = CombinedData.drop(['Unnamed: 0.1','Unnamed: 0'], axis='columns')
            

    # CombinedData.info()

    for i in CombinedData.columns:
        nans += printInfo(CombinedData[i],i,printthis=False)

print(nans)
#%%

#Read all files and make tensors to save them in seperated files:


def GetAndCombineAllSubs(folders,extenstion):

    raw_all = None  
    # lables_all = np.zeros((1,1))   
    # print(raw_all.shape)

    raw_all = []
    print(len(raw_all))
    for folder in folders:
            
                
        exportfoldername= "output"

        foldername= folder+"\\"+exportfoldername+"\\"

            
        filepath = foldername+folder+"_Combined"+extenstion
        print(filepath)
        CombinedData = pd.read_csv(filepath)
        CombinedData = CombinedData.drop(['Unnamed: 0.1','Unnamed: 0'], axis='columns')
        
        if len(raw_all) == 0:
            raw_all = CombinedData
        else:
            raw_all = np.concatenate((raw_all, CombinedData))
        # lables_all = np.concatenate((lables_all,lables))
    names= CombinedData.columns.values
    
    
    
    if not ThisisRandomSub:
        #creat a new continues timeline    
        timeline = np.linspace(0,len(raw_all)/100,len(raw_all))
        raw_all[:,-1] = timeline
    
    
    return raw_all , names


ThisisRandomSub = False
ThisisRandomSub = True
if ThisisRandomSub:
    subs = ["Sub8"]
    subs = ["RandomSub1"]
    extrationInfo = 's8'
    extrationInfo = 'rs1'
else:
    subs = ["Sub2","Sub3","Sub5","Sub6","Sub7","Sub8"]
    subs = ["Sub2","Sub3","Sub5","Sub6","Sub7"]
    extrationInfo = 's2s3s5s6s7s8'
    extrationInfo = 's2s3s5s6s7'

extenstion = "_Label.csv"
lable_all , lable_names = GetAndCombineAllSubs(subs,extenstion)
print(lable_names)
print(lable_all.shape)

# # plt.plot(raw_all[:,-1],raw_all[:,:-1])
# plt.plot(lable_all[:,-1],lable_all[:,:-1])
# plt.legend(lable_names)
# plt.title(extenstion)
# plt.show()

    

extenstion = "_Arm.csv"
signal_raw , signal_names = GetAndCombineAllSubs(subs,extenstion)
print(signal_names)
print(signal_raw.shape)

# print(np.min(signal_raw[:,:-1]))
# print(np.max(signal_raw[:,:-1]))

# plt.plot(signal_raw[:,-1],signal_raw[:,:-1])
# plt.legend(signal_names )
# plt.title(extenstion)
# plt.xlim((1300,1302))
# plt.show()


with open('CombinedData_Arm_'+extrationInfo+'.npy', 'wb') as f:
    np.save(f, signal_raw)
    # np.save(f, signal_names)
    np.save(f, lable_all)
    # np.save(f, lable_names)
    


extenstion = "_Head.csv"
signal_raw , signal_names = GetAndCombineAllSubs(subs,extenstion)
print(signal_names)
print(signal_raw.shape)

# plt.plot(signal_raw[:,-1],signal_raw[:,:-1])
# plt.legend(signal_names)
# plt.title(extenstion)
# plt.show()

with open('CombinedData_Head_'+extrationInfo+'.npy', 'wb') as f:
    np.save(f, signal_raw)
    # np.save(f, signal_names)
    np.save(f, lable_all)
    # np.save(f, lable_names)
    


extenstion = "_Hand.csv"
signal_raw , signal_names = GetAndCombineAllSubs(subs,extenstion)
print(signal_names)
print(signal_raw.shape)

# plt.plot(signal_raw[:,-1],signal_raw[:,:-1])
# plt.legend(signal_names)
# plt.title(extenstion)
# plt.show()


with open('CombinedData_Hand_'+extrationInfo+'.npy', 'wb') as f:
    np.save(f, signal_raw)
    # np.save(f, signal_names)
    np.save(f, lable_all)
    # np.save(f, lable_names)



#%%
print("dddddddd")



with open('CombinedData_Arm_rs1.npy', 'wb') as f:
    np.save(f, signal_raw)
    # np.save(f, signal_names)
    np.save(f, lable_all)
    # np.save(f, lable_names)


lable_names = pd.DataFrame(lable_names)
lable_names.to_csv("CombinedData_Arm_rs1")
    

#%%
print("open")
    
with open('CombinedData_Arm_rs1.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
    
print(a.shape)
print(b.shape)

