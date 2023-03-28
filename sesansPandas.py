#!/usr/bin/env python
# coding: utf-8
# Written by Wim Bouwman w.g.bouwman@tudelft.nl, Februari 2023
# Works only for finished scans with only one single empty beam measurement (labelled as empty beam)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from scipy.optimize import curve_fit
from ipywidgets import interact
import pandas as pd

def str2sec(timestr): # converts a single string describing a date and time into seconds after last century?
    timeobj = datetime.strptime(timestr, '%Y-%m-%d  %H:%M:%S')
    return timeobj.timestamp()

class SesansReductionC:
    def __init__(self):
        self.pars = ()
        self.raw = ()
        self.pols = ()
        self.empty = ()
        self.normalised = ()
        self.samples = ()

        # SesansReductionC.FileRead(self,file_number,paths)
        # SesansReductionC.DropRubbish(self)
        # SesansReductionC.FindNumberSinePoints(self)        
        # SesansReductionC.SelectGoodRaw(self)
        # SesansReductionC.Date2TimeStruc(self)
        # SesansReductionC.AddSpinEchoLength(self)
        # SesansReductionC.CalcYE(self)
        # SesansReductionC.MeanSines(self)
        # SesansReductionC.getpol(self,fitplot=True)
        # SesansReductionC.FindSamples(self)
        # SesansReductionC.GetEmpty(self)
        # SesansReductionC.Normalise(self)

    def FileRead(file_number,paths):
        self=SesansReductionC()
        #Scan reads in a data file from SESANS
        p = paths[0] #filepath to original sample data
        data_file = 'se'+"%06d" % (file_number,)+'.dat'
        raw_data = open(p+'/'+ data_file)
        data = raw_data.readlines() 

        #This part of the code finds the number of parameters (npar), number of columns (ncolumn), and number of points (npoints).
        #Since this info is given in the first three lines, it only loops over the first three lines in the data.
        A = []
        for line in data[:3]: 
            desc_data = re.split(r'[\t]', line)
            desc_array = np.array(desc_data)
            desc_values = desc_array[1]
            A.append(desc_values)
        npar = int(A[0]) 
        ncolumn = int(A[1])
        npoints = int(A[2])

        #Here we find the header names by looping only over every element in the line after the last parameter, 
        #which is where the header names are located.
        for par in data[npar:npar+1]:
            par_data = re.split(r'[\t]', par)
            header_names = np.array(par_data)

        #To find the header units we loop over every element in the line where the header units are located, which is after the header names line. 
        #Some units are empty since not all header names have units. An example of a unitless header name is the polarisation.
        for par in data[npar+1:npar+2]:
            par_data2 = re.split(r'[\t]', par)
            header_units = np.array(par_data2)

        #The data values are all after the parameter units line.
        #So we loop for every line after the units line until the end of the file to find all the data values.
        data_arrays = []
        for line in data[npar+2:]:
            data_lists = re.split(r'[\t]', line)
            data_lists[-1] = data_lists[-1][:-1] #Removes a whiteline character in the last value. 
            data_array_i = np.array(data_lists)
            data_arrays.append(data_array_i)
        data_arrays = np.array(data_arrays)

        #The parameter names & values are found by looping over every line in the file until the line where the last parameter is located.
        #The line where the last parameter is located is found using npar.
        #The parameter names are the first column, and the parameter values are the second column.
        par_names = ['filenumber']
        par_values = [file_number]
        for line in data[:npar]:
            par_data3 = re.split(r'[\t]', line)
            par_data_i = np.array(par_data3)
            par_names.append(par_data_i[0])
            par_values.append(par_data_i[1])

        par_names = np.array(par_names)
        par_values = np.array(par_values)
        header_names[-1] = header_names[-1][:-1] #Removes a whiteline character in the header names. 
        self.pars=pd.DataFrame(par_values, index=par_names)
        self.raw=pd.DataFrame(data_arrays,columns=header_names)
        return self

    def DropRubbish(self): # drops unnecessary data
        self.raw=self.raw.drop(['Matrix','FIELDVALUEM1 OFF','FIELDVALUEM2 OFF','FIELDVALUEM3 OFF','FIELDVALUEM4 OFF','XPOS OFF','MonOFF', 'Det1OFF', 'Det2OFF', 'SE length'], axis=1)

    def SelectGoodRaw(self): # cuts off an unfinished sine scan
        nsub=float(self.pars.loc["Nrsubsetpoints"].values[0])
        [nrow,ncol]=self.raw.shape
        #maxgood is up until where we can still find sine points.
        number_of_points = math.floor(nrow/nsub) 
        maxgood= int(number_of_points*nsub)
        self.raw=self.raw.iloc[:maxgood]        

    def Date2TimeStruc(self):
        StartTimeSec=str2sec(self.pars.loc["start date time"].values[0])
        TimeNames=["StartTOFF","StartTON"]
        for TimeName in TimeNames:
            if TimeName in self.raw.columns:
                TimeStrs=self.raw[TimeName].values
                time_after_start = []
                for i in TimeStrs:
                    time_after_start.append(str2sec(i)-StartTimeSec)
                self.raw.loc[:,TimeName]=time_after_start
        self.raw=self.raw.astype(float)

    def AddSpinEchoLength(self):
        B = self.raw["FIELDSETPOINT"].to_numpy() #mT
        c = 4.63e14 # T^-1 m^-2
        theta = float(self.pars.loc["tilt angle (°)"].values[0]) *np.pi/180. # convert from degrees to radians
        length = 1.340 # m
        lamb = float(self.pars.loc["wavelength detector 2 (nm)"].values[0]) *1.0e-9
        sel = 1.0e6*c*lamb*lamb*length*B*0.001/np.tan(theta)/np.pi # micrometre
        spin_echo_length = pd.DataFrame(sel, columns=["spin-echo length [um]"])
        self.raw=pd.concat([self.raw,spin_echo_length],axis=1)                                                                    

    def CalcYE(self):
        y = self.raw["Det0OFF"].to_numpy()
        e = np.sqrt(y)
        YE = pd.DataFrame(np.stack((y,e),axis=1), columns=["Y","E"])
        self.raw=pd.concat([self.raw,YE],axis=1)          

    def FindNumberSinePoints(self): # finds the number of points in a sine scan 
        x = self.raw["Guide field 2"].to_numpy()
        nsp=0       
        while x[nsp+1] > x[nsp]:   
            nsp=nsp+1
        self.pars.at["Nrsubsetpoints", 0] = str(nsp+1)

    def MeanSines(self):
        data=self.raw.to_numpy()
        nsub=int(self.pars.loc["Nrsubsetpoints"].values[0])
        [nrow,ncol]=data.shape
        nsin=int(nrow/nsub)
        data3D=data.reshape(nsin,nsub,ncol)
        datamean=np.mean(data3D,axis=1)
        self.pols = pd.DataFrame(datamean, columns = self.raw.columns)

    def getpol(self,fitplot=True):
        #Function on which we fit is defined here.
        def func(xx,ishim,a,delta):
            TT=T_start              
            return ishim*(1+ a*np.sin(2*np.pi*xx/TT-delta))

        #We split the values up into rows of subset points each.
        nsub=int(self.pars.loc["Nrsubsetpoints"].values[0])
        x = self.raw["Guide field 2"].to_numpy()
        y = self.raw["Y"].to_numpy()
        yer = self.raw["E"].to_numpy()
        nrow=len(x)
        nsin=int(nrow/nsub)
        x=x.reshape(nsin,nsub) 
        y=y.reshape(nsin,nsub) 
        yer=yer.reshape(nsin,nsub) 

        #We initialise arrays, which will be used later.
        pol = np.zeros(nsin)
        poler = np.zeros(nsin)
        shim = np.zeros(nsin)
        shimer = np.zeros(nsin)
        phase = np.zeros(nsin)
        phaseer = np.zeros(nsin)
        if int(self.pars.loc["filenumber"].values[0]) > 13622:
            T_start = 0.3843 # determined from se013622.dat March 2023 for longer wavelength
        else:
            T_start = 0.4136 # determined from se012833.dat April 2022, does not work for short spin echo length < 100 nm

        #Here we loop over every row of intensities.
        for i in range(nsin):
            xi = x[i]  
            yi = y[i]  
            yeri = yer[i]
            #We find where the max value of the intensities is. This is where the phase should be pi/2
            max_index =np.argmax(yi) 
            max_start =max(yi) 
            ishim_start = np.mean(yi)
            
            #Define start values
#            T_start = 0.3843 # determined from se013622.dat March 2023, does not work for short spin echo length < 100 nm
#            T_start = 0.4136 # determined from se012833.dat April 2022, does not work for short spin echo length < 100 nm
            pol_start=max_start/ishim_start - 1
            tmp= -0.5*np.pi + 2*np.pi*xi[max_index]/T_start
            delta_start=tmp - 2*np.pi*math.floor(tmp/2/np.pi)
            start_ = [ishim_start, pol_start, delta_start]

            #Curve_fit is used to find the best estimated values of the three parameters we fit with.
            bound = ((0, 0,-np.pi), (max_start, 2, np.pi*3)) # Fixing the phase too strict (0 .. pi) can give bad fits
            popt, pcov = curve_fit(func,xi,yi,sigma=yeri,bounds=bound,p0=start_)
            perr = np.sqrt(np.diag(pcov))
            shim[i]=popt[0] 
            pol[i]=popt[1]
            phase[i]=popt[2]
            shimer[i] = perr[0] 
            poler[i] = perr[1] 
            phaseer[i] = perr[2] 

        title = "3 fit parameter method"
        #Here we can plot yfit and the measured values if need be.
        if fitplot==True:
            def update_plot(i):
                y_fit = func(x[i],shim[i],pol[i],phase[i])
                plt.title(title)
                plt.errorbar(x[i],y[i],fmt='g.',xerr=0,yerr=yer[i],ecolor='r')
                plt.plot(x[i],y_fit,color='b')
                plt.xlabel('Guide field [A]')
                plt.ylabel('Intensity (norm.)')
            interact(update_plot, i=(0,len(shim)-1,1)) 

        poldata = np.transpose(np.array([pol, poler, shim, shimer, phase, phaseer]))
        pol_names =  ['polarisation', 'polarisation error', 'shim', 'shim error', 'phase', 'phase error']
        polframe = pd.DataFrame(poldata, columns = pol_names)
        self.pols=pd.concat([self.pols,polframe],axis=1)                                                                    

    def FindSamples(self): 
        SamplePos = np.around(self.pols["X_axis"].to_numpy())
        samples=np.unique(SamplePos) # unique sample positions
        number_of_samples = len(samples) #How many samples were used
        #Nincycle is used to check up until which sample number the sample array repeats (ex. for (4,6,9,4,6,9), nincycle =3 )
        nincycl=number_of_samples-1       
        while not np.any(SamplePos[0:nincycl] == SamplePos[nincycl:2*nincycl]):   
            nincycl=nincycl+1
        
        Position=np.around(SamplePos[:nincycl]).astype(int) #Here we take the used samples which were continuously cycled.
        SampleNames =[None] *nincycl
        SampleThick_cm= [None] *nincycl
        empty= [None] *nincycl
        for i in range(nincycl):
            SamplePosString = ("sample: Nr "+str(int(Position[i])))
            SampleNameThick = self.pars.loc[SamplePosString].values[0]
            tmp=str(SampleNameThick).split()
            SampleNames[i] = " ".join(tmp[:-1])
            SampleThick_cm[i] = float(tmp[-1])*0.1
            sl = np.char.lower(SampleNames[i])
            empty[i] =( np.any(np.char.find(sl,'air')>-1) or np.any(np.char.find(sl,'empty')>-1))
#            empty[i] =( np.any(np.char.find(sl,'air')>-1) or np.any(np.char.find(sl,'empty')>-1) or np.any(np.char.find(sl,'p0')>-1))
        SampleNames = np.array(SampleNames,dtype=object)
        SampleThick_cm = np.array(SampleThick_cm)
        empty = np.array(empty)
        self.samples = pd.DataFrame({"position": Position.astype(float), "name": SampleNames, "thickness[cm]": SampleThick_cm, "empty": empty})

    def GetEmpty(self): 
        EmptyPosition=self.samples[self.samples["empty"]]["position"].values[0]
        self.empty=self.pols[self.pols["X_axis"] == EmptyPosition].loc[:, ["spin-echo length [um]", "polarisation", "polarisation error", "shim", "shim error"]]

    def Normalise(self):
        NSamp=len(self.samples.index)
        Normalised_df=[]
        Shim=[]
        ShimEmpty=self.empty["shim"].mean()
        P0=self.empty["polarisation"].values[:]
        dP0=self.empty["polarisation error"].values[:]
        lamb = float(self.pars.loc["wavelength detector 2 (nm)"].values[0]) *10 # in \AAngstrom
        for i in range(NSamp):
            OneSample=self.pols[self.pols["X_axis"] == self.samples.loc[i,"position"]].loc[:, ["StartTOFF", "spin-echo length [um]", "polarisation", "polarisation error", "shim", "shim error"]]
            OneSample.reset_index(drop=True, inplace=True)
            Ps=OneSample["polarisation"].values[:]
            dPs=OneSample["polarisation error"].values[:]
            Pn=Ps/P0
            dPn = (1/P0)*dPs + (Ps/(2*P0**2))*dP0
            t = max(self.samples.loc[i,"thickness[cm]"],0.001) # to prevent divide by zero for empty beam
            G = np.log(Pn)/(lamb**2*t)
            dG = (1/Pn)*dPn/(lamb**2*t)
            NormSample=pd.DataFrame(np.stack((Pn,dPn,G,dG),axis=1), columns=["P/P0","P/P0 error","G","G error"])
            OneSample=pd.concat([OneSample,NormSample],axis=1)
            Normalised_df.append(OneSample)
            Shim.append(OneSample["shim"].mean())

        Shim=np.array(Shim)
        Transmission=Shim/ShimEmpty
        ShimTrans=pd.DataFrame(np.stack((Shim,Transmission),axis=1), columns=["shim","transmission"])
        self.samples=pd.concat([self.samples,ShimTrans],axis=1)
        self.normalised = Normalised_df

    def Analyse(file_number,paths):
        self=SesansReductionC.FileRead(file_number,paths)
        SesansReductionC.DropRubbish(self)
        SesansReductionC.FindNumberSinePoints(self)        
        SesansReductionC.SelectGoodRaw(self)
        SesansReductionC.Date2TimeStruc(self)
        SesansReductionC.AddSpinEchoLength(self)
        SesansReductionC.CalcYE(self)
        SesansReductionC.MeanSines(self)
        SesansReductionC.getpol(self,fitplot=True)
        SesansReductionC.FindSamples(self)
        SesansReductionC.GetEmpty(self)
        SesansReductionC.Normalise(self)
        return self

def ExcelWrite(self,paths):   
    fileout=paths[1]+'/'+ self.pars.loc["Data filename"].values[0].replace("dat", "xlsx")

    with pd.ExcelWriter(fileout) as writer:
        self.pars.to_excel(writer, sheet_name="parameters")
        self.raw.to_excel(writer, sheet_name="raw data")
        self.pols.to_excel(writer, sheet_name="polarisations")
        self.empty.to_excel(writer, sheet_name="empty beam")
        self.samples.to_excel(writer, sheet_name="samples")
        self.empty.to_excel(writer, sheet_name="empty beam")
        for i in range(len(self.samples.index)):
            self.normalised[i].to_excel(writer, sheet_name=self.samples.loc[i,"name"])
        
def SesWrite(self,paths): #path argument
    for i in range(len(self.samples.index)):
        if not self.samples.loc[i,"empty"]:
            fileout=paths[1]+'/'+ self.pars.loc["Data filename"].values[0].replace(".dat", "")+'_'+"%02.0f" % (i) + ".ses"
            t=self.samples.loc[i,"thickness[cm]"]
            tempheader = ('FileFormatVersion 1.0',
            'DataFileTitle '+str(self.pars.loc['Title'].values[0]).replace(' ','_'),
            'Sample '+self.samples.loc[i,"name"],
            'Thickness '+"{:.1f}".format(t*10),
            'Thickness_unit mm',
            'Theta_zmax 0.002',
            'Theta_zmax_unit radians',
            'Theta_ymax 0.002',
            'Theta_ymax_unit radians',
            'Orientation Z',
            'SpinEchoLength_unit A',
            'Depolarisation_unit A^{-2}cm^{-1}',
            'Wavelength_unit A',
            '','BEGIN_DATA',
            'SpinEchoLength Depolarisation Depolarisation_error Wavelength')
            echoLengths = self.normalised[i]["spin-echo length [um]"].values[:]*10**4
            lambdas = np.ones_like(echoLengths)*float(self.pars.loc["wavelength detector 2 (nm)"].values[0]) *10
            tempdata = np.empty([len(echoLengths),4])
            tempdata[:,0] = echoLengths
            tempdata[:,1] = self.normalised[i]["G"].values[:]
            tempdata[:,2] = self.normalised[i]["G error"].values[:]
            tempdata[:,3] = lambdas
            np.savetxt(fileout,tempdata,header='\n'.join(tempheader),comments='')

def NormPlot(self): #Plot of spin echo length against normalised polarisation
    plt.figure()
    for i in range(len(self.samples.index)):
        if not self.samples.loc[i,"empty"]:
            plt.errorbar(self.normalised[i]["spin-echo length [um]"].values[:],
            self.normalised[i]["P/P0"].values[:],
            fmt='.',
            label=self.samples.loc[i,"name"],
            yerr=self.normalised[i]["P/P0 error"].values[:])
    plt.xlabel('spin-echo length [$\mu m$]')
    plt.ylabel('$P/P_0$')   
    plt.legend()
    plt.xlim(left=0)
    # ylow = np.amax([0, np.amin(pn-en)])
    # plt.ylim(ylow,1.1)     
    plt.title(self.pars.loc["Data filename"].values[0])
    
def EmptyPlot(self): #Plot of spin echo length against normalised polarisation
    plt.figure()
    plt.errorbar(self.empty["spin-echo length [um]"].values[:],
    self.empty["polarisation"].values[:],
    fmt='.',
    yerr=self.empty["polarisation error"].values[:])
    plt.xlabel('spin-echo length [$\mu m$]')
    plt.ylabel('polarisation')   
    plt.xlim(left=0)
    plt.title(self.pars.loc["Data filename"].values[0])
    
def ShimPlot(self): #Plot of spin echo length against normalised polarisation
    plt.figure()
    for i in range(len(self.samples.index)):
        plt.errorbar(self.normalised[i]["spin-echo length [um]"].values[:],
        self.normalised[i]["shim"].values[:],
        fmt='.',
        label=self.samples.loc[i,"name"],
        yerr=self.normalised[i]["shim error"].values[:])
    plt.xlabel('spin-echo length [$\mu m$]')
    plt.ylabel('intensity')   
    plt.legend()
    plt.xlim(left=0)
    plt.title(self.pars.loc["Data filename"].values[0])
    


