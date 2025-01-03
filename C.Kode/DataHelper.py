# DataHelper.py
from genericpath import isfile
import glob
from ntpath import join
from os import listdir
import os
import numpy as np
import pandas as pd

from Backbone import Movement, Sensor, SensorData, Trigger    
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm  # Import tqdm for progress bar

class Dataloader:
    def __init__(self,showPrint= False):
        self.showPrint = showPrint
    
    def saveData(self, data, filename):
        data.to_csv(filename, index=False)
        
    
    def getFileFromMovement(self,folder="",movement= Movement.piano,filename = ""):
        #Switchcase for movement
        
        #Find all files in folder
        arg = os.listdir(folder)
        if movement == Movement.piano:
            
            #fild contain movement
            fileinfolder = [f for f in arg if 'piano' in f]
            # print(fileinfolder)
     
            
                        
            # file_path = 'TestMeaurement\sub_piano1__fs_99.0___acc_data_1970-01-01 00_10_40.csv'
            # file_path = folder+'sub_klaver 1__fs_99.0___acc_data_1970-01-01 00_13_47.csv'
            # file_path = folder+'sub_piano2__fs_99.0___acc_data_1970-01-01 02_37_53.csv'
            # file_path = folder+'sub_3piano__fs_99.0___acc_data_1970-01-01 00_05_20.csv'
            # file_path = folder+"\\"+fileinfolder
            # print(file_path)
        elif movement == Movement.fist:
            fileinfolder = [f for f in arg if 'fist' in f]
            
            # file_path = folder+'sub_fist 1__fs_99.0___acc_data_1970-01-01 00_16_54.csv'
            # file_path = folder+'sub_fist2__fs_99.0___acc_data_1970-01-01 02_48_40.csv'
            # file_path = folder+'sub_3fist__fs_100.0___acc_data_1970-01-01 00_09_18.csv'
        elif movement == Movement.grib:
            fileinfolder = [f for f in arg if 'grib' in f]
            
            # file_path = folder+'sub_grib1__fs_99.0___acc_data_1970-01-01 00_20_06.csv'
            # file_path = folder+'sub_grib2__fs_100.0___acc_data_1970-01-01 02_45_31.csv'
            # file_path = folder+'sub_3grib__fs_99.0___acc_data_1970-01-01 00_12_09.csv'
        elif movement == Movement.slag:
            fileinfolder = [f for f in arg if 'punch' in f]
            
            # file_path = folder+'sub_slag1__fs_99.0___acc_data_1970-01-01 00_23_13.csv'
            # file_path = folder+'sub_punch2__fs_100.0___acc_data_1970-01-01 02_42_36.csv'
            # file_path = folder+'sub_3punch__fs_99.0___acc_data_1970-01-01 00_16_48.csv'
        else:
            fileinfolder = [filename]
            print("Movement not found")
            # return None        
    
        
        file_path = folder+"\\"+fileinfolder[0]
        
        return file_path    
    
    def LoadDataFromFilePath(self,file_path):
        rawData = pd.read_csv(file_path)
        return rawData
    

    def GetInputAndControlData(self,rawData):
        headColumns = rawData.columns[rawData.columns.str.contains('Head')] 
        headData = rawData[headColumns]/16384.0
        
        armColumns = rawData.columns[rawData.columns.str.contains('Arm')] 
        armData = rawData[armColumns]/16384.0
        # print(armColumns)
        
        handColumns = rawData.columns[rawData.columns.str.contains('Hand')] 
        handData = rawData[handColumns]/16384.0
        # print(handColumns)
        

        #get all columns withouth "Head"
        controlData = rawData.drop(columns=headColumns)
        controlData = controlData.drop(columns='trigger')
        return headData,controlData,armData,handData
    
    def GetTrigger(self,rawData):
        trigger_signal = rawData['trigger']
        trigger_diff = trigger_signal.diff()
        trigger_count = trigger_diff[trigger_diff == 1].count()
        trigger_Idx = rawData[trigger_diff == 1].index
        
        
        #Create a trigger object to store the trigger signal
        trigger : Trigger = Trigger( trigger_signal,trigger_count,trigger_Idx,trigger_diff)
        return trigger

    def GetTimeline(self,signal,fs=100):
        
        N_sample = len(signal)
        T_step = 1/fs
        timeline_s = np.linspace(0.0, N_sample*T_step, N_sample)
        return timeline_s

    
    
    
    def getDataFromFilePath(self,file_path):     
        
        
        rawData = self.rawData(file_path)
                
        headColumns = rawData.columns[rawData.columns.str.contains('Head')] 
        inputData = rawData[headColumns]

        #get all columns withouth "Head"
        controlData = rawData.drop(columns=headColumns)
        controlData = controlData.drop(columns='trigger')


        trigger_signal = rawData['trigger']
        trigger_diff = trigger_signal.diff()
        trigger_count = trigger_diff[trigger_diff == 1].count()
        trigger_StartIdx = rawData[trigger_diff == 1].index
        
        #Create a trigger object to store the trigger signal
        trigger = {}
        trigger["signal"] = trigger_signal
        trigger["count"] = trigger_count
        trigger["StartIdx"] = trigger_StartIdx
        trigger["diff"] = trigger_diff
        
            
        self.trigger = trigger
        self.rawDatad = rawData
        self.inputData = inputData
        self.controlData = controlData

    def getData(self):
        return self.rawData,self.inputData,self.controlData,self.trigger
    
    
    def TotalrawData(self, file_path):
        # Load raw data
        rawData = pd.read_csv(file_path)
        return rawData
        
    def rawData(self, file_path):
        # Load raw data
        rawData = pd.read_csv(file_path)
        
        # # Extract triggers and identify trigger count
        # trigger_signal = rawData['trigger']
        # trigger_diff = trigger_signal.diff()
        # trigger_count = trigger_diff[trigger_diff == 1].count()
        # trigger_StartIdx = rawData[trigger_diff == 1].index
        
        # #Extrat each extrimity
        # Sensor1 = rawData.iloc[:, 1:4]
        # Sensor2 = rawData.iloc[:, 4:7]
        # Sensor3 = rawData.iloc[:, 7:10]
        # Sensor4 = rawData.iloc[:, 10:13]
        # Sensor5 = rawData.iloc[:, 14:17]
        # Sensor6 = rawData.iloc[:, 18:21]
        # #change name of columns
        # Sensor1.columns = ['X', 'Y', 'Z']
        # Sensor2.columns = ['X', 'Y', 'Z']
        # Sensor3.columns = ['X', 'Y', 'Z']
        # Sensor4.columns = ['X', 'Y', 'Z']
        # Sensor5.columns = ['X', 'Y', 'Z']
        # Sensor6.columns = ['X', 'Y', 'Z']
        
        # Sensor1.loc[:, 'Magnitude'] = np.sqrt((Sensor1**2).sum(axis=1))
        
        # Sensor2['Magnitude'] = np.linalg.norm(Sensor1, axis=1)

        
        
        # #calculate the magnitude of each sensor
        # Sensor1['Magnitude'] =  np.sqrt((Sensor1[['X', 'Y', 'Z']]**2).sum(axis=1))
        # Sensor2['Magnitude'] =  np.sqrt((Sensor2[['X', 'Y', 'Z']]**2).sum(axis=1))
        # Sensor3['Magnitude'] =  np.sqrt((Sensor3[['X', 'Y', 'Z']]**2).sum(axis=1))
        # Sensor4['Magnitude'] =  np.sqrt((Sensor4[['X', 'Y', 'Z']]**2).sum(axis=1))
        
        return rawData#,trigger_signal,trigger_count,trigger_StartIdx, Sensor1,Sensor2,Sensor3,Sensor4,Sensor5,Sensor6
    

    
    def splitData(self,Sensordata, trigger_StartIdx, window_size):
        # Split data
        data = []
        for idx in trigger_StartIdx:
            data.append(Sensordata.iloc[idx:idx+window_size])
        return data

        
    
    def load_data(self, file_path):
        
        try:           
                        
            #read file
            print("loading data", end='\r')
            data = self.load_csv(file_path)
            
            ch_data, names, ch_trigger, triggername,trigger_count = self.extract_data(data)

            if self.showPrint:
                self.presentRawData(file_path,ch_data, names, ch_trigger, triggername,trigger_count)
           
            
            #update printed text with new text
            print("Loading is done", end='\r')
       
            return ch_data, names, ch_trigger, triggername,trigger_count
            
            
            
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
        except pd.errors.ParserError:
            print("Error parsing the file.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def load_csv(self,file_path):
        try:           
                        
            file = open(file_path,"r")

            #read file
            data = file.readlines()
                        
            file.close()
            if self.showPrint:
                print(f"CSV file '{file_path}' loaded successfully.")            
            return data                      
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
        except pd.errors.ParserError:
            print("Error parsing the file.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def extract_data(self,data):
        #extract names
        names = data[0].split(",")
        if self.showPrint:
            print(names)
        
        #remove names from data  
        data = data[1:]

        #extract each column
        colums = list()

        for i in tqdm(range(0, len(data)), desc="Processing",mininterval = 0.5):
            #remove brackets
            data[i] = data[i].replace("[","")
            data[i] = data[i].replace("]","")
            #remove white space
            data[i] = data[i].replace(" ","")
            #split data
            data[i] = data[i].split(",")
            #convert to float
            data[i] = [float(x) for x in data[i]]
            #add to column
            for j in range(0, len(data[i])):
                if len(colums) <= j:
                    colums.append(list())
                colums[j].append(data[i][j])

        # subtract trigger column
        ch_trigger = colums[0]
        ch_data = colums[1:]
        triggername = names[0]
        names = names[1:]
        
        trigger_count = sum(1 for i in range(len(ch_trigger)-1) if ch_trigger[i] == 0 and ch_trigger[i+1] == 1)
        
        #convert to array
        ch_data = np.array(ch_data)
        
        return ch_data, names, ch_trigger, triggername,trigger_count
    
    
    def presentRawData(self,file_path,ch_data, names, ch_trigger, triggername,trigger_count):
        print(str(len(ch_trigger)) + " " + str(triggername))
        #print max and min of trigger
        print("max trigger: " + str(max(ch_trigger)) + " min trigger: " + str(min(ch_trigger)))
        print('Number of triggers:', trigger_count)
        print(f"CSV file '{file_path}' segmenteded successfully.")
            

        

class DataExtractor:
    def __init__(self,showPrint= False):
        self.showPrint = showPrint
        self.data = None
        
    
    def getMagnitude(self,x,y,z):
        return np.sqrt(x**2 + y**2 + z**2)
    
    
    
    def getSeparatedSensorData(self,mixedSensorData,Sensornames):
        
        SeparatedSensors : Sensor = []
        
        for sensorname in Sensornames:
            sensor = self.Extractsensors(mixedSensorData,sensorname)
            SeparatedSensors.append(sensor)
        
        return SeparatedSensors

    
    def Extractsensors(self,combineddata, sensorname):  
        SensoreDatanames = combineddata.columns[combineddata.columns.str.contains(sensorname)]
        Sensordata = combineddata[SensoreDatanames]
        #rename columns to x,y and z
        Sensordata.columns = ['x','y','z']
        sensordata : SensorData = SensorData( Sensordata.iloc[:,0],Sensordata.iloc[:,1],Sensordata.iloc[:,2])
        

        dataExtractor = DataExtractor()

        meg=  dataExtractor.getMagnitude(sensordata.X ,sensordata.Y ,sensordata.Z)

        filter = DataFiltre(showPrint=False)
        mag_filtered = filter.butter_bandpass_filter(meg, lowcut=1, highcut=25, fs=100, order=5)
       
        sensordata.Meg = meg
        sensordata.Meg_filtered = mag_filtered
        
        
        
        sensor : Sensor = Sensor(sensorname,sensordata)
        
        
        return sensor
    
    
        
        
        


from scipy import signal
from scipy.signal import butter, lfilter,filtfilt
class DataFiltre:
    
    def __init__(self,showPrint= False):
        if showPrint:
            print("DataFiltre") 
        

    def butter_bandpass(self,lowcut, highcut, fs, order=1):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(self,signal, lowcut=1, highcut=25, fs=100, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        # y = lfilter(b, a, signal)
        y = filtfilt(b, a, signal,padlen=0)
        
        return y
    
    def gethighpass(self,highcut, fs, order=1):
        # Design the low-pass filter
        nyq = 0.5 * fs
        normal_cutoff = highcut / nyq
        # print(normal_cutoff)

        return butter(order, normal_cutoff, btype='highpass')


    def HighPass_filter(self,signal, highcut=1, fs=100, order=5):
        b, a = self.gethighpass(highcut, fs, order=order)
  
        y = filtfilt(b, a, signal,padlen=5000)


        return y


    def getAutoThreshold(self,segment, window=100):
        segment = segment - segment.mean()
        # print(segment.head())
        Sensordata = segment
        Sensordatameg = (Sensordata**2)
        Sensordatameg = Sensordatameg.sum(axis=1)
        Sensordatameg = np.sqrt(Sensordatameg)
        
        lowcut = 1
        highcut = 25
        fs = 100
        y = self.butter_bandpass_filter(Sensordatameg, lowcut, highcut, fs, order=2)
        y = signal.medfilt(y, kernel_size=5)
        diff = np.diff(y)
        diff = diff**2
        diff = diff/np.max(diff)
        
        #moveing average
        diff = pd.Series(diff)
        diff = diff.rolling(window).mean()

        
        Sensordatameg = diff
        
        # Sensordatameg = Sensordatameg.rolling(window).mean()
            
        std = Sensordatameg.std()
        threshold = std
        return threshold, Sensordatameg

    
