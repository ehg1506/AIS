#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:52:25 2017

@author: jonleonhardsen
"""

import PlotVesselPatrick as PV
import datetime

Database = '/Users/PatrickAndreNaess/Desktop/ContainerFleet.db'
#Database ='/Users/erikgrundt/Desktop/ContainerFleet.db'
#Database = '/Volumes/LaCie/NTNUfilesMac/ContainerFleet.db'
ports = '/Users/PatrickAndreNaess/Documents/PYTHON/LINERLIB-master/data/ports.csv'

#All methods are run from this script, 1 = run
SegmentAnalysis = 1 #1 = analyze segment, 0 = analyze vessel
GlobalMap = 0
LocalMap = 0
DraughtInterpolate = 0
TradeDirectionsSuez = 0
TradeDirectionPacific = 0
FreightRateImpact = 0
MonthlyAverage = 0
GeoDistribution = 0
SpeedDev = 0
SpeedHistogram = 0
SpeedForAnalysis = 0
GeoAnalysis = 0
CumulatedSpeed = 0
DraughtAnalysis = 0
OrnsteinUhlenbeckGenerator = 0

ClusterPorts = 0
#WriteOutTable = 0 Denne er puttet inn i WorldOceanAnalysis
PortAnalysis = 0
Network = 0
WorldOceanAnalysis = 1 

BigShips = 1
Panamax = 0
MuSim = 0
FuelCalc = 0
FuelAgg = 0

Pacific = [-120,63,12,28]
Atlanter = [0,53,-80,33]
MexAu = [180,7.75,-180,-26.10]
Global = [180,90,-180,-90]
asEur = [120, 50, -15, 0]
CapeAfrica = [40,-20,10,-40]
Panama = [-79,9.8,-81,8.5]

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

#Always analyse from 01/xx/xxxx --> PolygonAnalysis
lowtime = '01/01/2013'
hightime = '01/01/2016'
maxspeed = 25
minspeed = 5

unixlow = datetime.datetime.strptime(lowtime, "%d/%m/%Y").timestamp()
unixhigh = datetime.datetime.strptime(hightime, "%d/%m/%Y").timestamp()
    
#PV calls PlotVessels.py, which essentially does all data extracting, initial analyses,
#and visualizations of AIS data.  
if SegmentAnalysis == 1:
    PV.ExtractData(Database,Pos[0],Pos[1],Pos[2],Pos[3],unixlow,unixhigh \
                   ,maxspeed,minspeed,SegmentAnalysis)
else:
    PV.ExtractData(Vessel,Pos[0],Pos[1],Pos[2],Pos[3])

if ClusterPorts == 1:
   clusterlat,clusterlon,Hcluster = PV.DataClusterPorts(ports)

if GlobalMap == 1:    
    PV.GlobalMap()

if DraughtInterpolate == 1:
    PV.DraughtInterpolate()    
if TradeDirectionsSuez == 1:
    PV.TradeDirectionSuez()
if TradeDirectionPacific == 1:
    PV.TradeDirectionPacific()
if MonthlyAverage == 1:
    PV.MonthlyAverage()
if FreightRateImpact == 1:
    PV.FreightRateImpact()
if GeoDistribution == 1:    
    PV.GeoDistribution()
if SpeedDev == 1:
    PV.SpeedDev()
if SpeedHistogram == 1:
    PV.SpeedHistogram()
if SpeedForAnalysis == 1:
    PV.SpeedForAnalysis()   
if GeoAnalysis == 1:
    minspeed = 12
    PV.GeoAnalysis(Pos[0],Pos[1],Pos[2],Pos[3],minspeed)
if CumulatedSpeed == 1:
    PV.CumulatedSpeed()    
if DraughtAnalysis == 1:
    PV.DraughtAnalysis()

    
if PortAnalysis == 1:
    portdata = PV.Checkports(ports)
#if WriteOutTable == 1:
#    BigD = PV.DataFrameForAnalysis()
if WorldOceanAnalysis == 1:
    BigD = PV.PolygonAnalysis(unixlow,unixhigh)
if Network == 1:
    route = PV.ShippingNetwork()
    
if LocalMap == 1:    
    PV.LocalMap()
    