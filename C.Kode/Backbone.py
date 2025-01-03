#Backbone.py
from enum import Enum

from matplotlib import pyplot as plt

 
class Movement(Enum):
    NA = 0,
    NoMovement = 1,
    piano = 2,
    fist = 3,
    grib = 4,
    slag = 5

class Lejring(Enum):
    NA = 0,
    BackSide = 1,
    RightSide = 2,
    LeftSide = 3
    

#Create sections from trigger to trigger
from typing import Tuple

class Trigger:
    signal = []
    movement = []
    count = 0
    Idx = []
    diff = []
    Duaration = 0
    timeline_s = None
    lejring = []
    
    
    def __init__(self,signal,count,Idx,diff):
        self.signal = signal
        self.count = count
        self.Idx = Idx
        self.diff = diff
        
    def print(self):
        print(f"Trigger count: {self.count}")
        # for i in range(0,self.count):
            # print(f"Idx: {self.Idx[i]}")
        if self.timeline_s is not None:
            print(f"length of signal in sek.: {self.timeline_s.max()}")
        # print(f"diff: {self.diff}")
    
    def plot(self,adjustedhight = True):
        data = self.signal
        adjustedData = data/data.max()
        plt.plot(adjustedData, label="trigger")
        
    def SetTimeline(self,timeline_s):
        self.timeline_s = timeline_s
        
    def SetMovementSiganl(self,movement):
        self.movement = movement
    
    def SetLejring(self,lejring):
        self.lejring = lejring

        
        
        
        
class SensorData:
    X = []
    Y = []
    Z = []
    Meg = []
    Meg_filtered = []
    
    def __init__(self,X,Y,Z):
        self.X = X
        self.Y = Y
        self.Z = Z
 
    def SetMegnitute(self,Meg,Meg_filtered):
        self.Meg = Meg
        self.Meg_filtered = Meg_filtered      
    
    def print(self,BeforePrint = ""):
        #print the shape of the 5 elements in opersit order
        print(f"{BeforePrint}{self.X.shape} : X "   + f"{self.Y.shape} : Y " + f"{self.Z.shape} : Z " + f"{self.Meg.shape} : Meg " + f"{self.Meg_filtered.shape} : Meg_filtered ")

class Sensor:
    name = ""
    data: SensorData = None
    
    def __init__(self,name=None,data =None):
        self.name = name
        self.data = data
        
    def print(self):
        print(f"Sensor: {self.name}")
        print(f"Data: {self.data.X.shape} : X "   + f"{self.data.Y.shape} : Y " + f"{self.data.Z.shape} : Z " + f"{self.data.Meg.shape} : Meg " + f"{self.data.Meg_filtered.shape} : Meg_filtered ")
        
    def SetMegnitute(self,Meg):
        self.data.Meg = Meg
    
    def SetfilteredMegnitute(self,Meg):
        self.data.Meg_filtered = Meg
        
    def plot(self,adjustedhight= True):
        data = self.data.Meg_filtered
        adjustedData = data/data.max()
        plt.plot(adjustedData, label=self.name)

        
        


class Section:
    def __init__(self,No= None,StartIndex= None,EndIndex= None,Duaration= None):
        self.No = No
        self.StartIndex = StartIndex
        self.EndIndex = EndIndex
        self.Duaration = Duaration
        self.controlData = [Sensor()]
        self.inputData = [Sensor()]
    No = 0
    StartIndex = 0
    EndIndex = 0
    Duaration = 0
    controlData: Sensor = []
    inputData: Sensor = []
    movement : Movement = None
    
    def print(self):
        print(f"Section {self.No} has duration {self.Duaration} sec")
        print(f"StartIndex: {self.StartIndex} EndIndex: {self.EndIndex}")
        print(f"Movement: {self.movement}")
        print(f"ControlData: {len(self.controlData)}")
        print(f"InputData: {len(self.inputData)}")
    

    def SetMovement(self,movement: Movement):
        self.movement = movement
        
    def AddControlData(self,controlData: Sensor):
        self.controlData.append(controlData)
    
    def addInputData(self,inputData: Sensor):
        self.inputData.append(inputData)
        
    def SeperateSectionData(self):
        for sensor in self.controlData:
            sensorData : Sensor = sensor
            sensorData.X = sensorData.data.X[self.StartIndex:self.EndIndex]
            sensorData.Y = sensorData.data.Y[self.StartIndex:self.EndIndex]
            sensorData.Z = sensorData.data.Z[self.StartIndex:self.EndIndex]
            sensorData.Meg = sensorData.data.Meg[self.StartIndex:self.EndIndex]
            sensorData.Meg_filtered = sensorData.data.Meg_filtered[self.StartIndex:self.EndIndex]
  
            