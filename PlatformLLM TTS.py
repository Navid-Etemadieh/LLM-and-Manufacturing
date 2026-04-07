from __future__ import division
import serial,collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from tkinter import messagebox as mbox
from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
from datetime import datetime
from time import time,sleep,ctime, localtime, strftime
import statistics
import scipy.fftpack as fourier
import numpy as np
import pandas as pd
import seaborn
import cProfile
from functools import partial
import csv
import os
from datetime import datetime
import threading
from tkinter import *
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import io
import re
import threading
from gtts import gTTS
import pygame
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

print(f"Python Processing Device: {device}")


pygame.mixer.init()

isReceive= False 
isRun = True 
second=False
t2=True
global accanim
global ampanim
global temanim
global ambanim
global acc_ax
global amp_ax
global tem_ax
global amb_ax
global thread
global avrg
global llmdata
points = 0.0
accXPoint=0.0
accYPoint=0.0
accZPoint=0.0
ampM1Point=0.0
ampM2Point=0.0
ampM3Point=0.0
ampMFPoint=0.0
temPoint=0.0
tembedPoint=0.0
ambhumPoint=0.0
ambtemPoint=0.0
sampleD = 100
maxframes=8000
timepoints = []
realTime=[]
realTimeImg=[]
temdata = []
tembeddata = []
ambhumdata = []
ambtemdata = []
accXdata= []
accYdata= []
accZdata= []
ampM1data= []
ampM2data= []
ampM3data= []
ampMFdata= []
data0 = []
savedData=[]
timePointsSaved=[]
accymin = -1
accymax = 1
ampymin = -3
ampymax = 3
temxmin = 0
temxmax = sampleD
temymin = 0
temymax = 250
ambymin = 0
ambymax = 60
start_time = 0
current_time=0
sampling=[]
Posm=0
localtime=[]
Fourier=np.empty(shape=0)
M_Fourier=np.empty(shape=0)
Fs=0
Fdata=[]
view_time = 4
Amax=0
Amin=0
Rep=1
capturingData=False
showAvrg=False
pause=False
finalx=0
Start=False
global arduino
ports=[]
port1=''

drive_path = r"C:\Users\pazuni01\OneDrive - University of Louisville\Research\AnomalyDetectionPlatform\arduino_data.csv"

#LLM
global model
model = OllamaLLM(model="llama-3-8b")
template = """
You are an expert in 3D printing and you are going to teach a beginner about the process.

Here is the question to answer: {instruction}
These are the values of the real-time data on the 3D printer: {input}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def main():
    global thread
    def puertos_seriales():
        global port1
        ports = ['COM%s' % (i + 1) for i in range(256)]
        portsFound = []
        first=True
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                portsFound.append(port)
                if first:
                    port1=port
                    first=False
            except (OSError, serial.SerialException):
                pass
        return portsFound
        if not os.path.exists(drive_path):
            with open(drive_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp','NozzleTemp [C]', 'AccelerationX [g]', 'AccelerationY [g]','AccelerationZ [g]', 'AmbientTemp [C]', 'AmbientHum [%]','CurrentX [A]', 'CurrentY [A]', 'CurrentZ [A]','CurrentFil [A]', 'BedTemp [C]', 'Filament'])
  

    def Iniciar():
        global timepoints
        global realTime
        global realTimeImg
        global temdata
        global tembeddata
        global ambhumdata
        global ambtemdata
        global accXdata
        global accYdata
        global accZdata
        global ampM1data
        global ampM2data
        global ampM3data
        global ampMFdata
        global data0
        global savedData
        global timePointsSaved
        global sampling
        global accanim
        global ampanim
        global temanim
        global ambanim
        global isRun
        global avrg
        global start_time
        global arduino
        global acc_ax
        global amp_ax
        global tem_ax
        global amb_ax
        timepoints = []
        realTime=[]
        realTimeImg=[]
        accXdata = []
        accYdata = []
        accZdata = []
        ampM1data = []
        ampM2data = []
        ampM3data = []
        ampMFdata = []
        temdata = []
        tembeddata = []
        ambhumdata = []
        ambtemdata = []
        data0 = []
        savedData=[]
        timePointsSaved=[]
        sampling=[]
        

        try:
            arduino = serial.Serial(portUsed.get(), 9600, timeout=1)
            isReceiving = True
            isRun = True   
            thread.start()
            bStart['state'] = DISABLED

            #Acceleration Plot
            accfig = plt.figure(facecolor="0.55",figsize=(fw*2, fh), clear=True)
            accfig.patch.set_facecolor('#FFFFFF')
            plt.title("Acceleration",color='black',size=tf)
            plt.xticks([],fontsize=tickf)
            plt.yticks([],fontsize=tickf)
            acc_ax = plt.axes(xlim=(0,view_time),ylim=(accymin,accymax))
            acc_ax.set_xlabel("Time [sec]",fontsize=lf)
            acc_ax.set_ylabel("Acceleration [g]",fontsize=lf) 
            linesX = acc_ax.plot([] ,[], 'r',label="X axis")[0]
            linesY = acc_ax.plot([] ,[], 'g',label="Y axis")[0]
            linesZ = acc_ax.plot([] ,[], 'b',label="Z axis")[0]
            plt.xlim(0,view_time)
            plt.legend(loc='upper left')
            plt.tight_layout()
            #avrg = acc_ax.axhline(y=0, color = 'r',alpha=0,linewidth=0.5,linestyle='--')

            accplot = FigureCanvasTkAgg(accfig, master = graphic1,)
            accplot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
            accplot.draw()    
            #timepoints.append(time()-start_time)
            accanim = animation.FuncAnimation(accfig, aPlotData, fargs=(sampleD, linesX, linesY, linesZ), interval=33, blit=False,frames=maxframes)

            #Current Plot Settings
            ampfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
            ampfig.patch.set_facecolor('#FFFFFF')
            plt.title("Motor Current",color='black',size=tf)
            plt.xticks([],fontsize=tickf)
            plt.yticks([],fontsize=tickf)
            amp_ax = plt.axes(xlim=(0,view_time),ylim=(ampymin,ampymax))
            amp_ax.set_xlabel("Time [sec]",fontsize=lf)
            amp_ax.set_ylabel("Current [A]",fontsize=lf)
            linesM1 = amp_ax.plot([] ,[], 'r',label="Motor Z")[0]
            linesM2 = amp_ax.plot([] ,[], 'g',label="Motor Y")[0]
            linesM3 = amp_ax.plot([] ,[], 'b',label="Motor X")[0]
            linesMF = amp_ax.plot([] ,[], 'black',label="Motor Filament")[0]
            plt.xlim(0,view_time)
            plt.legend(loc='upper left')
            plt.tight_layout()

            ampplot = FigureCanvasTkAgg(ampfig, master = graphic2,)
            ampplot._tkcanvas.grid(row = 0,column = 1, padx = 0,pady = 0,sticky="N")
            ampplot.draw()    
            #timepoints.append(time()-start_time)
            ampanim = animation.FuncAnimation(ampfig, ampPlotData, fargs=(sampleD, linesM1, linesM2, linesM3,linesMF), interval=33, blit=False,frames=maxframes)


            #Temperature Plot
            temfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
            temfig.patch.set_facecolor('#FFFFFF')
            plt.title("Temperature",color='black',size=tf)
            plt.xticks([],fontsize=tickf)
            plt.yticks([],fontsize=tickf)
            tem_ax = plt.axes(xlim=(0,view_time),ylim=(temymin,temymax))
            tem_ax.set_xlabel("Time [sec]",fontsize=lf)
            tem_ax.set_ylabel("Temperature [°C]",fontsize=lf)
            linestemp = tem_ax.plot([] ,[], 'b',label="Nozzle")[0]
            linestempbed = tem_ax.plot([] ,[], 'g',label="Bed")[0]
            plt.xlim(0,view_time)
            plt.legend(loc='upper left')
            plt.tight_layout()
            #avrg = tem_ax.axhline(y=0, color = 'r',alpha=0,linewidth=0.5,linestyle='--')

            templot = FigureCanvasTkAgg(temfig, master = graphic2,)
            templot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
            templot.draw()    
            #timepoints.append(time()-start_time)
            temanim = animation.FuncAnimation(temfig, tPlotData,  fargs=(sampleD,linestemp,linestempbed),interval = 33, blit = False,frames=maxframes )

        except:
            try:
                arduino = serial.Serial(portUsed.get(), 9600, timeout=1)
                isReceiving = True
                isRun = True   
                thread.start()
                bStart['state'] = DISABLED

                #Acceleration Plot
                accfig = plt.figure(facecolor="0.55",figsize=(fw*2, fh), clear=True)
                accfig.patch.set_facecolor('#FFFFFF')
                plt.title("Acceleration",color='black',size=tf)
                plt.xticks([],fontsize=tickf)
                plt.yticks([],fontsize=tickf)
                acc_ax = plt.axes(xlim=(0,view_time),ylim=(accymin,accymax))
                acc_ax.set_xlabel("Time [sec]",fontsize=lf)
                acc_ax.set_ylabel("Acceleration [g]",fontsize=lf) 
                linesX = acc_ax.plot([] ,[], 'r',label="X axis")[0]
                linesY = acc_ax.plot([] ,[], 'g',label="Y axis")[0]
                linesZ = acc_ax.plot([] ,[], 'b',label="Z axis")[0]
                plt.xlim(0,view_time)
                plt.legend(loc='upper left')
                plt.tight_layout()
                
                accplot = FigureCanvasTkAgg(accfig, master = graphic1,)
                accplot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
                accplot.draw()    
                #timepoints.append(time()-start_time)
                accanim = animation.FuncAnimation(accfig, aPlotData, fargs=(sampleD, acc_ax, linesX, linesY, linesZ), interval=33, blit=False,frames=maxframes)

                #Current Plot Settings
                ampfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
                ampfig.patch.set_facecolor('#FFFFFF')
                plt.title("Motor Current",color='black',size=tf)
                plt.xticks([],fontsize=tickf)
                plt.yticks([],fontsize=tickf)
                amp_ax = plt.axes(xlim=(0,view_time),ylim=(ampymin,ampymax))
                amp_ax.set_xlabel("Time [sec]",fontsize=lf)
                amp_ax.set_ylabel("Current [A]",fontsize=lf)
                linesM1 = amp_ax.plot([] ,[], 'r',label="Motor Z")[0]
                linesM2 = amp_ax.plot([] ,[], 'g',label="Motor Y")[0]
                linesM3 = amp_ax.plot([] ,[], 'b',label="Motor X")[0]
                linesMF = amp_ax.plot([] ,[], 'black',label="Motor Filament")[0]
                plt.xlim(0,view_time)
                plt.legend(loc='upper left')
                plt.tight_layout()

                ampplot = FigureCanvasTkAgg(ampfig, master = graphic2,)
                ampplot._tkcanvas.grid(row = 0,column = 1, padx = 0,pady = 0,sticky="N")
                ampplot.draw()    
                #timepoints.append(time()-start_time)
                ampanim = animation.FuncAnimation(ampfig, ampPlotData, fargs=(sampleD, linesM1, linesM2, linesM3,linesMF), interval=33, blit=False,frames=maxframes)

                #Temperature Plot
                temfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
                temfig.patch.set_facecolor('#FFFFFF')
                plt.title("Temperature",color='black',size=tf)
                plt.xticks([],fontsize=tickf)
                plt.yticks([],fontsize=tickf)
                tem_ax = plt.axes(xlim=(0,view_time),ylim=(temymin,temymax))
                tem_ax.set_xlabel("Time [sec]",fontsize=lf)
                tem_ax.set_ylabel("Temperature [°C]",fontsize=lf)
                linestemp = tem_ax.plot([] ,[], 'b',label="Nozzle")[0]
                linestempbed = tem_ax.plot([] ,[], 'g',label="Bed")[0]
                plt.xlim(0,view_time)
                plt.legend(loc='upper left')
                plt.tight_layout()
                #avrg = tem_ax.axhline(y=0, color = 'r',alpha=0,linewidth=0.5,linestyle='--')

                templot = FigureCanvasTkAgg(temfig, master = graphic2,)
                templot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
                templot.draw()    
                #timepoints.append(time()-start_time)
                temanim = animation.FuncAnimation(temfig, tPlotData,  fargs=(sampleD,linestemp,linestempbed),interval = 33, blit = True,frames=maxframes )

            except:
                mbox.showerror(title="Error connecting to port", message="Verify if the board is connected.")
                print("Error connecting to port")
                bStart['state'] = NORMAL



    def pointsA():
        global isRun
        global isReceive
        global accXPoint
        global accYPoint
        global accZPoint
        global ampM1Point
        global ampM2Point
        global ampM3Point
        global ampMFPoint
        global temPoint
        global tembedPoint
        global ambhumPoint
        global ambtemPoint
        global points
        global accXdata
        global accYdata
        global accZdata
        global ampM1data
        global ampM2data
        global ampM3data
        global ampMFdata
        global temdata
        global tembeddata
        global ambhumdata
        global ambtemdata
        global timepoints
        global capturingData
        global Start
        global llmdata
        #arduino.reset_input_buffer()
        i=0
        j=0 
        flag1=True

        while (isRun):
            if Start==False:
                start_time = time()
                Start=True
            if j==0:
                accXdata.append(accXPoint)
                accYdata.append(accYPoint)
                accZdata.append(accZPoint)
                ampM1data.append(ampM1Point)
                ampM2data.append(ampM2Point)
                ampM3data.append(ampM3Point)
                ampMFdata.append(ampMFPoint)
                temdata.append(temPoint)
                tembeddata.append(tembedPoint)
                ambhumdata.append(ambhumPoint)
                ambtemdata.append(ambtemPoint)
                timepoints.append(round(time()-start_time,2))
            #    if isReceive:
            #        sampling.append(timepoints[-1]-timepoints[-2])
                j=0
            else:
                j=j-1
            while (arduino.inWaiting()==0):
                pass

            points = arduino.readline()  # Decode and remove extra characters
            points = points.decode('utf-8', errors='ignore').strip()
            #print(points)
            values = points.split(",")
            if len(values) > 10:
                try:
                    temPoint = float(values[0])
                    accXPoint = float(values[1])
                    accYPoint = float(values[2])
                    accZPoint = float(values[3])
                    ambtemPoint = float(values[4])
                    ambhumPoint = float(values[5])
                    ampM1Point = float(values[6])
                    ampM2Point = float(values[7])
                    ampM3Point = float(values[8])
                    ampMFPoint = float(values[9])
                    tembedPoint = float(values[10])
                    #points = float(points)  # Convert to float
                    if flag1:
                        accXdata=[]
                        accYdata=[]
                        accZdata=[]
                        ampM1data=[]
                        ampM2data=[]
                        ampM3data=[]
                        ampMFdata=[]
                        temdata=[]
                        tembeddata=[]
                        ambhumdata=[]
                        ambtemdata=[]
                        start_time = time()
                        timepoints=[]
                        accXdata.append(accXPoint)
                        accYdata.append(accYPoint)
                        accZdata.append(accZPoint)
                        ampM1data.append(ampM1Point)
                        ampM2data.append(ampM2Point)
                        ampM3data.append(ampM3Point)
                        ampMFdata.append(ampMFPoint)
                        temdata.append(temPoint)
                        tembeddata.append(tembedPoint)
                        ambhumdata.append(ambhumPoint)
                        ambtemdata.append(ambtemPoint)
                        start_time = time()
                        timepoints.append(round(time()-start_time,2))
                        flag1=False
                    llmdata=[float(values[0]),float(values[1]),float(values[2]),float(values[3]),float(values[4]),
                            float(values[5]),float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10])]
                except ValueError:
                    print(f"Received invalid data: {points}")
            with open(drive_path, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time()
                dt = datetime.fromtimestamp(timestamp)
                formatted_time = f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
                writer.writerow([formatted_time] + values)
            isReceive = True  

    thread = Thread(target = pointsA) 

#FUNCTIONS       
    def aPlotData(self,sampleD,linesX,linesY,linesZ):
        global points 
        global temdata
        global ambhumdata
        global ambtemdata
        global accXdata
        global accYdata
        global accZdata
        global ampM1data
        global ampM2data
        global ampM3data
        global second
        global Rep
        global realTime
        global timePointsSaved
        global savedData
        global localtime
        global showAvrg
        global start_time
        global Start
        global sampling
        global current_time
        global acc_ax

        t2=True
        current_time = timepoints[-1]
        linesX.set_data(timepoints, accXdata) 
        linesY.set_data(timepoints, accYdata) 
        linesZ.set_data(timepoints, accZdata) 
        if pause==False:
            if current_time > view_time:
                acc_ax.set_xlim([current_time-view_time,current_time])
        second=True
        bLeft['state'] = NORMAL
        bRight['state'] = NORMAL
        bZoomIn['state'] = NORMAL
        bZoomOut['state'] = NORMAL
        bStop['state'] = NORMAL

    def ampPlotData(self,sampleD,linesM1,linesM2,linesM3,linesMF):
        global timepoints
        global points 
        global temdata
        global ambhumdata
        global ambtemdata
        global accXdata
        global accYdata
        global accZdata
        global ampM1data
        global ampM2data
        global ampM3data
        global ampMFdata
        global second
        global Rep
        global realTime
        global timePointsSaved
        global savedData
        global localtime
        global showAvrg
        global start_time
        global Start
        global sampling
        global amp_ax
        global current_time
        t2=True
        #print(timepoints[-1],temPoint)
        #current_time = timepoints[-1]
        if pause==False:
            if current_time > view_time:
                amp_ax.set_xlim([current_time-view_time,current_time])
        linesM1.set_data(timepoints, ampM1data) 
        linesM2.set_data(timepoints, ampM2data) 
        linesM3.set_data(timepoints, ampM3data)
        linesMF.set_data(timepoints, ampMFdata) 
        second=True
        bLeft['state'] = NORMAL
        bRight['state'] = NORMAL
        bZoomIn['state'] = NORMAL
        bZoomOut['state'] = NORMAL
        bStop['state'] = NORMAL

    def tPlotData(self,sampleD,linestemp,linestempbed):
        global points 
        global temdata
        global tembeddata
        global ambhumdata
        global ambtemdata
        global second
        global Rep
        global realTime
        global timePointsSaved
        global savedData
        global localtime
        global showAvrg
        global start_time
        global Start
        global sampling
        global tem_ax
        global current_time
        t2=True
        #print(timepoints[-1],temPoint)
        #current_time = timepoints[-1]
        if pause==False:
            if current_time > view_time:
                tem_ax.set_xlim([current_time-view_time,current_time])
        linestemp.set_data(timepoints, temdata)
        linestempbed.set_data(timepoints, tembeddata)  
        second=True
        bLeft['state'] = NORMAL
        bRight['state'] = NORMAL
        bZoomIn['state'] = NORMAL
        bZoomOut['state'] = NORMAL
        bStop['state'] = NORMAL

    def exit():
        global isRun
        global Start
        isRun = False 
        if Start:
            thread.join()
            arduino.close()
        root.destroy()
        root.quit()
    def stop():  
        global isRun
        global isReceiving 
        global temanim
        global ambanim
        global accanim
        global thread
        isRun = False 
        isReceiving = False
        accanim.event_source.stop()
        ampanim.event_source.stop()
        temanim.event_source.stop()
        ambanim.event_source.stop()
        thread.join()
        bStart['state'] = NORMAL
        try:
            arduino.close()
        except:
            arduino.close()
        thread = Thread(target = pointsA) 
        points=00.0
        temdata=[]
        ambhumdata=[]
        ambtemdata=[]
        sampling=[]
        
    def startSampling():
        global capturingData
        bSave['state'] = NORMAL
        bSampling['state'] = DISABLED
        capturingData=True
        menuFile.entryconfig(2,label="Save the data", state=NORMAL )

    def saveSampling():
        global realTime
        global timePointsSaved
        global savedData
        global localtime
        for i in range(0,len(realTime)):
            localtime.append(ctime(realTime[i]))
        timepointsc=[]
        for i in range(0,len(timePointsSaved)):
            timepointsc.append("%.6f" % timePointsSaved[i])
            timepointsc[i]=float(timepointsc[i])
        CSV = pd.DataFrame([savedData,timepointsc,localtime])
        CSV=CSV.T
        date=localtime(realTime[-1])
        CSV.columns=['Temperature [°C]', 'Time[s]','Date and time']
        title='Reporte '+str("%02d" %date.tm_hour)+'H'+str("%02d" %date.tm_min)+' '+str(date.tm_mday)+'-'+str(date.tm_mon)+'-'+str(date.tm_year)+'.csv'
        CSV.to_csv(title, sep=';',float_format='%',decimal=',')
        savedData=[]
        timePointsSaved=[]
        localtime=[]
        bSave['state'] = DISABLED
        menuFile.entryconfig(2,label="Save the data", state=DISABLED )

    def GridON():
        plt.grid()
        menuPlot.entryconfig(0,label="Enable grid",command=GridOFF)
    def GridOFF():
        plt.grid()
        menuPlot.entryconfig(0,label="Disable grid",command=GridON)

    def ZoomIN():
        global view_time
        global finalx
        view_time=view_time/2
        if pause:
            acc_ax.set_xlim([finalx-view_time,finalx])
            amp_ax.set_xlim([finalx-view_time,finalx])
            tem_ax.set_xlim([finalx-view_time,finalx])

    def ZoomOUT():
        global view_time
        global finalx
        view_time=view_time*1.5
        if pause:
            acc_ax.set_xlim([finalx-view_time,finalx])
            amp_ax.set_xlim([finalx-view_time,finalx])
            tem_ax.set_xlim([finalx-view_time,finalx])

    def showMean():
        global showAvrg
        global avrg
        showAvrg=True
        avrg.set_alpha(1)
        menuPlot.entryconfig(3,label="Hide mean value",command=hideMean)

    def hideMean():
        global showAvrg
        global avrg
        showAvrg=False
        avrg.set_alpha(0)
        menuPlot.entryconfig(3,label="Show mean value",command=showMean)

    def pausePlot():
        global pause 
        global finalx
        global start_time
        global timepoints
        pause=True
        finalx=timepoints[-1]
        acc_ax.set_xlim([finalx-view_time,finalx])
        amp_ax.set_xlim([finalx-view_time,finalx])
        tem_ax.set_xlim([finalx-view_time,finalx])
        menuPlot.entryconfig(4,label="Play",command=playPlot)
        bStop['text'] = "▶"
        bStop['command'] = playPlot
        bLeft['state'] = NORMAL
        bRight['state'] = NORMAL

    def playPlot():
        global pause 
        pause=False
        menuPlot.entryconfig(4,label="pause",command=pausePlot)
        bStop['text'] = "||"
        bStop['command'] = pausePlot
        bLeft['state'] = DISABLED
        bRight['state'] = DISABLED

    def leftFunc():
        global finalx
        global view_time
        finalx=finalx-view_time/20
        acc_ax.set_xlim([finalx-view_time,finalx])
        amp_ax.set_xlim([finalx-view_time,finalx])
        tem_ax.set_xlim([finalx-view_time,finalx])


    def rightFunc():
        global finalx
        global view_time
        finalx=finalx+view_time/20
        acc_ax.set_xlim([finalx-view_time,finalx])
        amp_ax.set_xlim([finalx-view_time,finalx])
        tem_ax.set_xlim([finalx-view_time,finalx])

    fw = 5 
    fh = 3
    tf = fw*3
    lf = fw*2
    tickf = fw*2.5

#Acceleration Plot Settings
    accfig = plt.figure(facecolor="0.55",figsize=(fw*2, fh), clear=True)
    accfig.patch.set_facecolor('#FFFFFF')
    plt.title("Acceleration",color='black',size=tf)
    plt.xticks([],fontsize=tickf)
    plt.yticks([],fontsize=tickf)
    acc_ax = plt.axes(xlim=(0,view_time),ylim=(accymin,accymax))
    acc_ax.set_xlabel("Time [sec]",fontsize=lf)
    acc_ax.set_ylabel("Acceleration [g]",fontsize=lf)
    linesX = acc_ax.plot([] ,[], 'r',label="X axis")[0]
    linesY = acc_ax.plot([] ,[], 'g',label="Y axis")[0]
    linesZ = acc_ax.plot([] ,[], 'b',label="Z axis")[0]
    plt.xlim(0,view_time)
    plt.legend(loc='upper left')
    plt.tight_layout()

#Current Plot Settings
    ampfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
    ampfig.patch.set_facecolor('#FFFFFF')
    plt.title("Motor Current",color='black',size=tf)
    plt.xticks([],fontsize=tickf)
    plt.yticks([],fontsize=tickf)
    amp_ax = plt.axes(xlim=(0,view_time),ylim=(ampymin,ampymax))
    amp_ax.set_xlabel("Time [sec]",fontsize=lf)
    amp_ax.set_ylabel("Current [A]",fontsize=lf)
    linesM1 = amp_ax.plot([] ,[], 'r',label="Motor Z")[0]
    linesM2 = amp_ax.plot([] ,[], 'g',label="Motor Y")[0]
    linesM3 = amp_ax.plot([] ,[], 'b',label="Motor X")[0]
    linesMF = amp_ax.plot([] ,[], 'black',label="Motor Filament")[0]
    plt.xlim(0,view_time)
    plt.legend(loc='upper left')
    plt.tight_layout()

#Temperature Plot Settings
    temfig = plt.figure(facecolor="0.55",figsize=(fw, fh), clear=True)
    temfig.patch.set_facecolor('#FFFFFF')
    plt.title("Temperature",color='black',size=tf)
    plt.xticks([],fontsize=tickf)
    plt.yticks([],fontsize=tickf)
    tem_ax = plt.axes(xlim=(0,view_time),ylim=(temymin,temymax))
    tem_ax.set_xlabel("Time [sec]",fontsize=lf)
    tem_ax.set_ylabel("Temperature [°C]",fontsize=lf)
    linestemp = tem_ax.plot([] ,[], 'b',label="Nozzle")[0]
    linestempbed = tem_ax.plot([] ,[], 'g',label="Bed")[0]
    plt.xlim(0,view_time)
    plt.legend(loc='upper left')
    plt.tight_layout()

#ROOT WINDOW
    root = Toplevel()
    root.protocol("WM_DELATE_WINDOW",exit)
    root.config(bg = "#FFFFFF")
    root.title("  \t\t\t\t Anomaly Detection Platform")
    root.state('zoomed')    
    root.resizable(0,0)
    #root.iconbitmap('icono.ico')

    #COLUMN 0
    #HEADER
    header = Frame(root, width = 300,height = 200, bg = "#AD0000")
    header.grid(row = 0,column = 0, padx = 20,pady =0)
    header.grid_propagate(False)
    header.config(relief = "sunken")
    header.config(cursor = "")
    #CHATBOX
    chat_frame = Frame(root, width=300, height=700, bg="#AD0000")
    chat_frame.grid(row=1, column=0, padx=20, pady=(5, 20))
    chat_frame.grid_propagate(False)
    chat_frame.config(relief="sunken")
    chat_text = Text(
        chat_frame,
        wrap=WORD,
        state=DISABLED,
        bg="#FFFFFF",
        font=("CairoliClassic 25 bold", 10)
    )
    chat_text.place(x=0, y=0, width=280, height=500)
    scrollbar = Scrollbar(chat_frame, command=chat_text.yview)
    scrollbar.place(x=280, y=0, height=500)
    chat_text.config(yscrollcommand=scrollbar.set)
    chat_text.tag_config("user", foreground="blue")
    chat_text.tag_config("bot", foreground="green")
    chat_text.tag_config("bold", font=("Arial", 10, "bold"))
    chat_text.tag_config("italic", font=("Arial", 10, "italic"))
    chat_text.tag_config("code", font=("Courier", 10), background="#EEEEEE")
    chat_text.tag_config("math", foreground="#AA00AA")
    user_entry = Entry(chat_frame, font=("Arial", 11))
    user_entry.place(x=5, y=510, width=210, height=30)
    send_button = Button(chat_frame, text="Send")
    send_button.place(x=220, y=510, width=60, height=30)

    def speak_text(text):
        try:
            # Strip out markdown symbols so the TTS doesn't read "asterisk asterisk bold asterisk asterisk"
            clean_text = re.sub(r'[*`$]', '', text)
            
            # Generate the speech
            tts = gTTS(text=clean_text, lang='en', slow=False)
            
            # Save to memory instead of a file
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Load and play the audio
            pygame.mixer.music.load(fp, 'mp3')
            pygame.mixer.music.play()
            
            # Keep the thread alive while the audio is playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"Audio playback error: {e}")

    def insert_formatted_text(text_widget, message):
        text_widget.config(state=NORMAL)
        patterns = [
            (r"\*\*(.*?)\*\*", "bold"),        # **bold**
            (r"\*(.*?)\*", "italic"),          # *italic*
            (r"`(.*?)`", "code"),              # `code`
            (r"\$\$(.*?)\$\$", "math"),        # $$math$$
            (r"\$(.*?)\$", "math"),            # $math$
        ]
        idx = 0
        while idx < len(message):
            match_obj = None
            match_start = len(message)
            for pattern, tag in patterns:
                match = re.search(pattern, message[idx:])
                if match:
                    start = idx + match.start()
                    if start < match_start:
                        match_start = start
                        match_obj = (match, tag)
            if match_obj:
                match, tag = match_obj
                start = idx + match.start()
                end = idx + match.end()
                text_widget.insert(END, message[idx:start])
                content = match.group(1)
                text_widget.insert(END, content, tag)
                idx = end
            else:
                text_widget.insert(END, message[idx:])
                break
        text_widget.insert(END, "\n")
        text_widget.config(state=DISABLED)
        text_widget.see(END)
   
    def add_message(sender, message):
        chat_text.config(state=NORMAL)

        if sender == "user":
            chat_text.insert(END, "You: " + message + "\n", "user")
        else:
            chat_text.insert(END, "AI: ", "bot")
            insert_formatted_text(chat_text, message)

        chat_text.config(state=DISABLED)
        chat_text.see(END)

    def ask_llm(question):
        global llmdata
        try:
            printer_data = llmdata
            # Get the LLM response
            result = chain.invoke({
                "instruction": question,
                "input": str(printer_data)
            })
            # 1. Update the UI with the text
            add_message("bot", result)
            # 2. Play the audio (this runs in the background thread, so the UI won't freeze!)
            speak_text(result)
        except Exception as e:
            add_message("bot", f"Error: {e}")

    def send_message(event=None):
        msg = user_entry.get().strip()
        if msg == "":
            return

        add_message("user", msg)
        user_entry.delete(0, END)

        # Show thinking indicator
        add_message("bot", "Thinking...")

        # Run LLM (and TTS) in background
        threading.Thread(target=ask_llm, args=(msg,), daemon=True).start()

    send_button.config(command=send_message)
    user_entry.bind("<Return>", send_message)   

    #COLUMN 1
    #HEADER
    header = Frame(root, width = 1150,height = 150, bg = "#FFFFFF")
    header.grid(row = 0,column = 1, padx = 20,pady =0)
    header.grid_propagate(False)
    header.config(relief = "sunken")
    header.config(cursor = "")
    Label(header, text="LLM-ASSISTED 3D-PRINTING MONITORING",font="CairoliClassic 25 bold",justify="center",bg = "#FFFFFF",fg="black").grid(pady=20,padx=100, row=0, column=2)
    Label(header, text="PORT:",font="CairoliClassic 12 ",justify=LEFT,bg = "#FFFFFF",fg="black").grid(pady=3,padx=0, row=1, column=0)
    portUsed=StringVar()
    portCOM=ttk.Combobox(header,width=13, textvariable=portUsed,font="CairoliClassic 12 ",justify="left",state="readonly",values=puertos_seriales())
    portCOM.grid(pady=0,padx=0, row=1, column=1)
    portCOM.set(port1)

    #PLOTS
    graphic = Frame(root, width = 1150,height = 1000, bg = "#FFFFFF")
    graphic.grid(row = 1,column = 1, padx = 0,pady = 0)
    graphic1 = Frame(graphic, width = 1000,height = 7000, bg = "#FFFFFF")
    graphic1.grid(row = 0,column = 0, padx = 0,pady = 0)
    accplot = FigureCanvasTkAgg(accfig, master = graphic1,)
    accplot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
    accplot.draw()
    graphic2 = Frame(graphic, width = 1000,height = 300, bg = "#FFFFFF")
    graphic2.grid(row = 1,column = 0, padx = 0,pady = 0)
    templot = FigureCanvasTkAgg(temfig, master = graphic2,)
    templot._tkcanvas.grid(row = 0,column = 0, padx = 0,pady = 0,sticky="N")
    templot.draw()
    ampplot = FigureCanvasTkAgg(ampfig, master = graphic2,)
    ampplot._tkcanvas.grid(row = 0,column = 1, padx = 0,pady = 0,sticky="N")
    ampplot.draw()
    #ambplot = FigureCanvasTkAgg(ambfig, master = graphic2,)
    #ambplot._tkcanvas.grid(row = 0,column = 1, padx = 0,pady = 0,sticky="N")
    #ambplot.draw()      

    #PLOT CONTROLS
    controls = Frame(root, width = 1000,height = 100, bg = "#FFFFFF")
    controls.grid(row = 2,column = 1, padx = 0,pady = 0)
    bZoomOut = Button(controls,command= ZoomOUT, text= "     -🔍︎",bg="#C4C4C4",fg="black", font="CairoliClassic 10  ",width=4,justify="center",state=DISABLED)
    bZoomOut.grid(row=0,column=0, padx=5,pady=2)
    bLeft = Button(controls,command= leftFunc, text= "←",bg="#C4C4C4",fg="black", font="CairoliClassic 10 bold ",width=4,justify="center",state=DISABLED)
    bLeft.grid(row=0,column=1, padx=5,pady=2)
    bStop = Button(controls,command= pausePlot, text= "||",bg="#C4C4C4",fg="black", font="CairoliClassic 10 bold ",width=4,justify="center",state=DISABLED)
    bStop.grid(row=0,column=2, padx=5,pady=2)
    bRight = Button(controls,command= rightFunc, text= "→",bg="#C4C4C4",fg="black", font="CairoliClassic 10 bold ",width=4,justify="center",state=DISABLED)
    bRight.grid(row=0,column=3, padx=5,pady=2)
    bZoomIn = Button(controls,command= ZoomIN, text= "🔍+",bg="#C4C4C4",fg="black", font="CairoliClassic 10  ",width=4,justify="center",state=DISABLED)
    bZoomIn.grid(row=0,column=4, padx=5,pady=2)

    #COLUMN 2
    #LOGO
    logo = Frame(root, width = 300,height = 200, bg = "#AD0000")
    logo.grid(row = 0,column = 2, padx = 0,pady = 0)

    #FRAME
    frame = Frame(root, width = 300,height = 700, bg = "#AD0000")
    frame.grid(row = 1,column = 2, padx = 0,pady = 0)
    frame.grid_propagate(False)
    frame.config(relief = "sunken")
    frame.config(cursor = "")

    bStart = Button(frame,command= Iniciar, text= "START",bg="#9A1220",fg="white", font="CairoliClassic 11 ",width=13,justify="center")
    bStart.grid(row=1,column=0, padx=55,pady=5)
    #BTerminar = Button(frame,command= stop, text= "FINALIZAR",bg="#0C376B",fg="white", font="CairoliClassic 11 ",width=13,state=DISABLED)
    #BTerminar.grid(row=2,column=0, padx=5,pady=5)
    bSampling = Button(frame,command= startSampling, text= "RECORD",bg="#9A1220",fg="white", font="CairoliClassic 11 ",width=13,justify="center",state=DISABLED)
    bSampling.grid(row=2,column=0, padx=5,pady=5)
    bSave = Button(frame,command= saveSampling, text= "STOP\nRECORDING",bg="#9A1220",fg="white", font="CairoliClassic 11 ",width=13,justify="center",state=DISABLED)
    bSave.grid(row=3,column=0, padx=5,pady=5)


    exit = Button(frame,command= exit, width=11 ,text= "EXIT",bg="#9A1220", fg="white",font="CairoliClassic 12 bold",justify="center")
    exit.grid(row=10,column=0, padx=5,pady=30)

    frame2 = Frame(root, width = 300,height = 100, bg = "#AD0000")
    frame2.grid(row = 2,column = 2, padx = 100,pady = 0)
    frame2.grid_propagate(False)
    frame2.config(relief = "sunken")
    frame2.config(cursor = "")

    #MENUS
    menuBar=Menu(root)
    root.config(menu=menuBar)
    menuFile = Menu(menuBar,tearoff=0)
    menuPlot = Menu(menuBar,tearoff=0)
    menuBar.add_cascade(label="File", menu=menuFile)
    menuBar.add_cascade(label="Plot", menu=menuPlot)

    menuFile.add_command(label="Start capturing data", command=Iniciar)
    menuFile.add_command(label="Start saving data", command=startSampling,state=DISABLED)
    menuFile.add_command(label="Save the data collected", command=saveSampling,state=DISABLED)
    menuFile.add_separator()
    menuFile.add_command(label="Exit", command=exit)

    menuPlot.add_command(label="Grid ON", command=GridON,state=DISABLED)
    menuPlot.add_command(label="Zoom in", command=ZoomIN)
    menuPlot.add_command(label="Zoom out", command=ZoomOUT)
    menuPlot.add_command(label="pause", command=pausePlot)

    root.mainloop()

if __name__ == '__main__':
    main()
