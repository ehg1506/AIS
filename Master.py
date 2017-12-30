#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:41:00 2017

@author: PatrickAndreNaess
"""

import AIS_Analysis as AIS
import datetime

Database = '/Users/PatrickAndreNaess/Desktop/ContainerFleet.db'
#Database ='/Users/erikgrundt/Desktop/ContainerFleet.db'
#Database = '/Volumes/LaCie/NTNUfilesMac/ContainerFleet.db'
ports = '/Users/PatrickAndreNaess/Documents/PYTHON/LINERLIB-master/data/ports.csv'

#All methods are run from this script, 1 = run
Analysis = 1 #1 = extract data of intrest from database
LocalMap = 0 # Plot all data on map 
DataClusterPorts = 0 # Cluster ports
PortAnalysis = 0 # Compare with LINNERLIB
GeoFencePorts = 1 # Geofence ports
Network = 1 # Network of port-to-port
WorldOceanAnalysis = 0 # Polygon
SpeedForAnalysis = 0 # Message interval
DraughtAnalysis = 0
DraughtHistogram = 0 # Draught histogram

Pacific = [-120,63,12,28]
Atlanter = [0,53,-80,33]
MexAu = [180,7.75,-180,-26.10]
Global = [180,90,-180,-90]
asEur = [120, 50, -15, 0]
CapeAfrica = [60,24,-30,-42]
Panama = [-79,9.8,-81,8.5]
Suez = [40,40,30,27]

#Choose location of intrest:
Loc = 4

if Loc == 1:
    Pos = Pacific
elif Loc == 2:
    Pos = Atlanter
elif Loc == 3:
    Pos = MexAu
elif Loc == 4:
    Pos = Global
elif Loc == 5:
    Pos = asEur
elif Loc == 6:
    Pos = CapeAfrica
elif Loc == 7:
    Pos = Panama
elif Loc == 8:
    Pos = Suez    

#Time window of intrest:
lowtime = '01/01/2011'
hightime = '01/01/2016'

#Speed of interst:
maxspeed = 5
minspeed = 0

#Convert from date to unixtime:
unixlow = datetime.datetime.strptime(lowtime, "%d/%m/%Y").timestamp()
unixhigh = datetime.datetime.strptime(hightime, "%d/%m/%Y").timestamp()
    
#PV calls PlotVessels.py, which essentially does all data extracting, initial analyses,
#and visualizations of AIS data.  
if Analysis == 1:
    plotlon,plotlat = AIS.ExtractData(Database,Pos[0],Pos[1],Pos[2],Pos[3],
                   unixlow,unixhigh,maxspeed,minspeed,Analysis)

if DataClusterPorts == 1:
   clusterlat,clusterlon,labels = AIS.ClusterPorts() 
   
if SpeedForAnalysis == 1:
    AIS.SpeedForAnalysis() 
    
if DraughtAnalysis == 1:
    AIS.DraughtAnalysis()
    
if PortAnalysis == 1:
    portdata = AIS.Checkports(ports)
    
if WorldOceanAnalysis == 1:
    BigD = AIS.PolygonAnalysis(unixlow,unixhigh)
    
if GeoFencePorts == 1:
    labels = AIS.GeoFencePorts()
    
if Network == 1:
    route,G = AIS.ShippingNetwork(labels)
    
if LocalMap == 1:    
    AIS.LocalMap()