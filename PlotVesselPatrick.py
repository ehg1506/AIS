#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:45:53 2017

@author: PatrickAndreNaess
"""


import sqlite3
import matplotlib.pyplot as plt


from netCDF4 import Dataset, num2date
import time
import datetime
from mpl_toolkits.basemap import Basemap
import urllib, os
import numpy as np
from scipy import stats
import statistics as st
import pandas
import pandas as pd
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as hcluster
import loc_check as LC

# create new figure, axes instances

def ExtractData(filepath,a,b,c,d,lowtime,hightime,maxspeed,minspeed,SegmentAnalysis):
    global draughttime
    global useridDraught
    global draught
    global plotlat
    global plotlon
    global speeds
    global geospeed
    global valspeed
    global timestep
    global mmsi
    global navstat
    
    speeds = list()
    plotlat = list()
    plotlon = list()
    valspeed = list()
    #lats = list()
    #lons = list()
    #unistep = list()
    timestep = list()
    geospeed = list()
    mmsi = list()
    navstat = list()

    conn = sqlite3.connect(filepath)
    cur = conn.cursor()
    
    # For nav-igation status, insert: 
    #portstatus = 'and (nav_status==1 or nav_status ==5)'
    
    SQLstring = "select distinct unixtime,sog,latitude,longitude,userid,\
        nav_status from %s where longitude<= %s and latitude <= %s\
        and longitude >= %s and latitude >= %s and sog >= %s and \
        sog <= %s and unixtime >= %s and unixtime <= %s %s" % ('PanamaxSample1',\
        str(a),str(b),str(c),str(d),str(minspeed),str(maxspeed),str(lowtime),\
        str(hightime),'')
    
    with conn:
        cur = conn.cursor()  
        if SegmentAnalysis == 1:
            cur.execute(SQLstring)
        else:
            cur.execute("SELECT unixtime,sog,latitude,longitude,userid FROM MessageType1 ORDER BY UNIXTIME ASC")
        VesselData = cur.fetchall()
        
        #####################################################
        # THE FOLLOWING PARTS FILTERS ON SPEED AND POSITION #
        #####################################################
        for i in range(0,len(VesselData)):
            Datastrip = VesselData[i]
            timestep.append(Datastrip[0])
            speeds.append(Datastrip[1])
            plotlat.append(Datastrip[2])
            plotlon.append(Datastrip[3])        
            mmsi.append(Datastrip[4])
            navstat.append(Datastrip[5])
            
    cur.close()
    '''
    conn = sqlite3.connect(filepath)
    cur = conn.cursor()
    with conn:
        cur = conn.cursor()
        if SegmentAnalysis == 1:
            cur.execute("select unixtime,draught,userid from BigShips5 order by userid, unixtime asc")
        else:
            cur.execute("SELECT unixtime,draught,userid FROM MessageType5 ORDER BY UNIXTIME ASC")
        draughtdata = cur.fetchall()
        draughttime = list()
        draught = list()
        useridDraught = list()
        
    for i in range(0,len(draughtdata)):
        draughtstrip = draughtdata[i]
        if Speed < 30 and lat <= maxLat and lat >= minLat \
            and lon >= minLon and lon <= maxLon and untime > lowtime and untime < hightime:
            if draughtstrip[1]/10 > 5:
                draughttime.append(draughtstrip[0])
                draught.append(draughtstrip[1]/10)
                useridDraught.append(draughtstrip[2])
    '''
    #return plotlat, plotlon

def DataFrameForAnalysis():
    # Lager df for analyse            
    df = pandas.DataFrame({'Speed': speeds, 'MMSI': mmsi, 'Unixtime': timestep, \
                             'Lat': plotlat, 'Lon': plotlon, 'Nav status': navstat})
    df['Date/Time'] = df['Unixtime'].apply(lambda x: \
                      datetime.fromtimestamp(x).strftime("%d/%m/%Y %H:%M:%S"))
    return df

def DataClusterPorts():
    global clusterlat
    global clusterlon
    global portnumber
    global Hcluster
    global k
    points = np.stack((plotlon,plotlat),axis=-1)
    N = len(points)
    
    #Cluster with a Hierarchical clustering algorithm to get k:
    thresh = 1
    Hcluster = hcluster.fclusterdata(points, thresh, criterion="distance")
    portnumber = pd.Series(Hcluster)
    k = portnumber.nunique()
    print('Found ' + str(k) + ' ports, from ' + str(N) + ' messages.')
    #Clustering with K-means algorithm:
    k_means = cluster.KMeans(n_clusters = k).fit(points).cluster_centers_
    clusterlat = k_means[:,0]
    clusterlon = k_means[:,1]
    
    return clusterlat,clusterlon,Hcluster
    
    
def Checkports(file):
    #Importing port data from LINERBIB:
    portdata = pd.DataFrame(pd.read_csv(file,sep="\t"))
    portdata['In cluster?'] = np.zeros(len(portdata))
    #Filtering out NANs from lon and lat:
    portdata = portdata[np.isfinite(portdata['Latitude']) & \
                        np.isfinite(portdata['Longitude'])]
    portdata.drop(['Draft','D_Region','Draft','CostPerFULL','CostPerFULLTrnsf', \
                   'PortCallCostFixed','PortCallCostPerFFE'], axis=1, inplace=True)    
    
    for i in range(0,len(clusterlat)):   
        for index, row in portdata.iterrows():
            if clusterlat[i] < (row['Latitude']+1) and \
               clusterlat[i] > (row['Latitude']-1) \
                and clusterlon[i] < (row['Longitude']+1) and \
                    clusterlon[i] > (row['Longitude']-1):
                    portdata['In cluster?'][index] += 1  
    
    portdata = portdata[portdata['In cluster?']>0]
    
    #Check integrity of portdata vs. cluster:                 
    if portdata['In cluster?'].sum() != k:
        print('ERROR: Found ' + str(portdata['In cluster?'].sum()) + \
              ' ports to ' + str(k) + ' clusters')
    
    portdata = portdata.nlargest(len(portdata),'In cluster?')
    
    return portdata
    
def ShippingNetwork():
    global route
    MMSI = pd.Series(mmsi).unique()
    route = list()
    
    for ship in MMSI:
        r = list()
        for i in range(0,len(mmsi)):
            if mmsi[i] == ship:
                if not r:
                    r.append(Hcluster[i])
                else:
                    if Hcluster[i] != Hcluster[i-1]:
                        r.append(Hcluster[i])
        route.append(r)
        
    return route           
                
            
def PolygonAnalysis():
    polygons = LC.generate_polygons()
    #LC.ocean_polygon(polygons)
    
    global number_of_vessel_monthly_atlantic 

    #Polygons indexing:
    # 0: Atlantic
    AtlanticMMSI = list()
    AtlanticSpeed = list()
    AtlanticLat = list()
    AtlanticLon = list()
    AtlanticTime = list()
    # 1: East Pacific
    EastPacificMMSI = list()
    EastPacificSpeed = list()
    EastPacificLat  = list()
    EastPacificLon = list()
    EastPacficTime = list()
    # 2: West Pacific
    WestPacificMMSI = list()
    WestPacificSpeed = list()
    WestPacificLat = list()
    WestPacificLon = list()
    WestPacificTime = list()
    # 3: Indian Ocean
    IndianOceanMMSI = list()
    IndianOceanSpeed = list()
    IndianOceanLat = list()
    IndianOceanLon = list()
    IndianOceanTime = list()
    # 4: Mediterranean
    MediterraneanMMSI = list()
    MediterraneanSpeed = list()
    MediterraneanLat = list()
    MediterraneanLon = list()
    MediterraneanTime = list()
    # 5: North Sea
    NorthSeaMMSI = list()
    NorthSeaSpeed = list()
    NorthSeaLat = list()
    NorthSeaLon = list()
    NorthSeaTime = list()
    # 6: SE Asia
    SEAsiaMMSI = list()
    SEAsiaSpeed = list()
    SEAsiaLat = list()
    SEAsiaLon = list()
    SEAsiaTime = list()
    # 7: Oceania
    OceaniaMMSI = list()
    OceaniaSpeed = list()
    OceaniaLat = list()
    OceaniaLon = list()
    OceaniaTime = list()
    
    #Outside
    MMSIOutsideZones = list()
    LatOutsideZones = list()
    LonOutsideZones = list()
    
    for i in range(0,len(plotlat)):
        if LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[0]):
            AtlanticMMSI.append(mmsi[i])
            AtlanticSpeed.append(speeds[i])
            AtlanticLat.append(plotlat[i])
            AtlanticLon.append(plotlon[i])
            AtlanticTime.append(timestep[i])
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[1]):
            EastPacificMMSI.append(mmsi[i])
            EastPacificSpeed.append(speeds[i])
            EastPacificLat.append(plotlat[i])
            EastPacificLon.append(plotlon[i])
            EastPacficTime.append(timestep[i])
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[2]):
            WestPacificMMSI.append(mmsi[i])
            WestPacificSpeed.append(speeds[i])
            WestPacificLat.append(plotlat[i])
            WestPacificLon.append(plotlon[i])
            WestPacificTime.append(timestep[i])   
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[3]):
            IndianOceanMMSI.append(mmsi[i])
            IndianOceanSpeed.append(speeds[i])
            IndianOceanLat.append(plotlat[i])
            IndianOceanLon.append(plotlon[i])
            IndianOceanTime.append(timestep[i]) 
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[4]):
            MediterraneanMMSI.append(mmsi[i])
            MediterraneanSpeed.append(speeds[i])
            MediterraneanLat.append(plotlat[i])
            MediterraneanLon.append(plotlon[i])
            MediterraneanTime.append(timestep[i]) 
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[5]):  
            NorthSeaMMSI.append(mmsi[i])
            NorthSeaSpeed.append(speeds[i])
            NorthSeaLat.append(plotlat[i])
            NorthSeaLon.append(plotlon[i])
            NorthSeaTime.append(timestep[i]) 
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[6]):
            SEAsiaMMSI.append(mmsi[i])
            SEAsiaSpeed.append(speeds[i])
            SEAsiaLat.append(plotlat[i])
            SEAsiaLon.append(plotlon[i])
            SEAsiaTime.append(timestep[i]) 
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[7]): 
            OceaniaMMSI.append(mmsi[i])
            OceaniaSpeed.append(speeds[i])
            OceaniaLat.append(plotlat[i])
            OceaniaLon.append(plotlon[i])
            OceaniaTime.append(timestep[i])
        else: 
            MMSIOutsideZones.append(mmsi[i])
            LatOutsideZones.append(plotlat[i])
            LonOutsideZones.append(plotlon[i])

    #Filters unique vessels
    Unique_vessels_Atlantic= list()
    Unique_vessels_EastPacific = list()
    Unique_vessels_WestPacific = list()
    Unique_vessels_IndianOcean = list()
    Unique_vessels_Mediterranean = list()
    Unique_vessels_NorthSea = list()
    Unique_vessels_SEAsia = list()
    Unique_vessels_Oceania = list()
    

    #Finding unique vessels for the whole period, function in Loccheck
    LC.find_unique_vessels(AtlanticMMSI,Unique_vessels_Atlantic)
    LC.find_unique_vessels(EastPacificMMSI,Unique_vessels_EastPacific)
    LC.find_unique_vessels(WestPacificMMSI,Unique_vessels_WestPacific)
    LC.find_unique_vessels(IndianOceanMMSI,Unique_vessels_IndianOcean)
    LC.find_unique_vessels(MediterraneanMMSI,Unique_vessels_Mediterranean)
    LC.find_unique_vessels(NorthSeaMMSI,Unique_vessels_NorthSea)
    LC.find_unique_vessels(SEAsiaMMSI,Unique_vessels_SEAsia)
    LC.find_unique_vessels(OceaniaMMSI,Unique_vessels_Oceania)

    print(len(Unique_vessels_Atlantic))
    
    #Finding the timeintervalls for all the different vessels, function in LocCheck
    # Denne må endres til en generisk timestamp
    Timestamps_Atlantic = LC.get_timevector(AtlanticTime)
    Timestamps_EastPacific = LC.get_timevector(EastPacficTime)
    Timestamps_WestPacfic = LC.get_timevector(WestPacificTime)
    Timestamps_IndianOcean = LC.get_timevector(IndianOceanTime)
    Timestamps_Medittarnean = LC.get_timevector(MediterraneanTime)
    Timestamps_NorthSea = LC.get_timevector(NorthSeaTime)
    Timestamps_SEAsia = LC.get_timevector(SEAsiaTime)
    Timestamps_Oceania = LC.get_timevector(OceaniaTime)
    #Hardcoding in the time here
    mintime = 1325376000
    maxtime = 1420070400
    minmaxtime = (1325376000,1420070400)
    
    Timestamps = LC.get_timevector(minmaxtime)
    
    print(Timestamps_Atlantic)
    print(Timestamps_EastPacific)
    print(Timestamps_WestPacfic)
    print(Timestamps_IndianOcean)
    print(Timestamps_Medittarnean)
    print(Timestamps_SEAsia)
    print(Timestamps_Oceania)
    
    #Generating monthly lists (will be a list of lists)
    number_of_messages_monthly_Atlantic = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_EastPacific = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_WestPacific = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_IndianOcean = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_Medittarnean = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_NorthSea = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_SEAsia = [[] for i in range(0,len(Timestamps))]
    number_of_messages_monthly_Oceania = [[] for i in range(0,len(Timestamps))] 
    #Filter data for monthly basis
    LC.monthly_filter(AtlanticTime,AtlanticMMSI,Timestamps,number_of_messages_monthly_Atlantic)
    LC.monthly_filter(EastPacficTime,EastPacificMMSI,Timestamps,number_of_messages_monthly_EastPacific)
    LC.monthly_filter(WestPacificTime,WestPacificMMSI,Timestamps,number_of_messages_monthly_WestPacific)
    LC.monthly_filter(IndianOceanTime,IndianOceanMMSI,Timestamps,number_of_messages_monthly_IndianOcean)
    LC.monthly_filter(MediterraneanTime,MediterraneanMMSI,Timestamps,number_of_messages_monthly_Medittarnean)
    LC.monthly_filter(NorthSeaTime,NorthSeaMMSI,Timestamps,number_of_messages_monthly_NorthSea)
    LC.monthly_filter(SEAsiaTime,SEAsiaMMSI,Timestamps,number_of_messages_monthly_SEAsia)
    LC.monthly_filter(OceaniaTime,OceaniaMMSI,Timestamps,number_of_messages_monthly_Oceania)
    print(len((number_of_messages_monthly_Atlantic[0])))
    #Make lists containing the number of unqiue vessel found in a time interval of one month
#    number_of_messages_monthly_Atlantic = number_of_messages_monthly_Atlantic[:-1]
#    number_of_messages_monthly_EastPacific = number_of_messages_monthly_EastPacific[:-1]
#    number_of_messages_monthly_WestPacific = number_of_messages_monthly_WestPacific[:-1]
#    number_of_messages_monthly_IndianOcean = number_of_messages_monthly_IndianOcean[:-1]
#    number_of_messages_monthly_Medittarnean = number_of_messages_monthly_IndianOcean[:-1]
#    number_of_messages_monthly_NorthSea = number_of_messages_monthly_NorthSea[:-1]
#    number_of_messages_monthly_SEAsia = number_of_messages_monthly_SEAsia[:-1]
#    number_of_messages_monthly_Oceania = number_of_messages_monthly_Oceania[:-1]
    #Generating list for each polygon
    unique_vessels_monthly_Atlantic = [[] for i in range(0,len(number_of_messages_monthly_Atlantic))]
    unique_vessels_monthly_EastPacific = [[] for i in range(0,len(number_of_messages_monthly_EastPacific))]
    unique_vessels_monthly_WestPacific = [[] for i in range(0,len(number_of_messages_monthly_WestPacific))]
    unique_vessels_monthly_IndianOcean = [[] for i in range(0,len(number_of_messages_monthly_IndianOcean))]
    unique_vessels_monthly_Medittarnean = [[] for i in range(0,len(number_of_messages_monthly_Medittarnean))]
    unique_vessels_monthly_NorthSea = [[] for i in range(0,len(number_of_messages_monthly_NorthSea))]
    unique_vessels_monthly_SEAsia = [[] for i in range(0,len(number_of_messages_monthly_SEAsia))]
    unique_vessels_monthly_Oceania = [[] for i in range(0,len(number_of_messages_monthly_Oceania))]
    #Filter unique vessels
    unique_vessels_monthly_Atlantic = LC.unique_vessels_monthly(number_of_messages_monthly_Atlantic,unique_vessels_monthly_Atlantic)
    unique_vessels_monthly_EastPacific = LC.unique_vessels_monthly(number_of_messages_monthly_EastPacific,unique_vessels_monthly_EastPacific)
    unique_vessels_monthly_WestPacific = LC.unique_vessels_monthly(number_of_messages_monthly_WestPacific,unique_vessels_monthly_WestPacific)
    unique_vessels_monthly_IndianOcean = LC.unique_vessels_monthly(number_of_messages_monthly_IndianOcean,unique_vessels_monthly_IndianOcean)
    unique_vessels_monthly_Medittarnean = LC.unique_vessels_monthly(number_of_messages_monthly_Medittarnean,unique_vessels_monthly_Medittarnean)
    unique_vessels_monthly_NorthSea = LC.unique_vessels_monthly(number_of_messages_monthly_NorthSea,unique_vessels_monthly_NorthSea)
    unique_vessels_monthly_SEAsia = LC.unique_vessels_monthly(number_of_messages_monthly_SEAsia,unique_vessels_monthly_SEAsia)
    unique_vessels_monthly_Oceania = LC.unique_vessels_monthly(number_of_messages_monthly_Oceania,unique_vessels_monthly_Oceania)
    print(unique_vessels_monthly_Atlantic)
    print(unique_vessels_monthly_EastPacific)
    print(unique_vessels_monthly_WestPacific)
    print(unique_vessels_monthly_IndianOcean)
    print(unique_vessels_monthly_Medittarnean)
    print(unique_vessels_monthly_NorthSea)
    print(unique_vessels_monthly_SEAsia)
    print(unique_vessels_monthly_Oceania)
    print(len(unique_vessels_monthly_Atlantic))
    print(len(unique_vessels_monthly_IndianOcean))
    print(len(unique_vessels_monthly_NorthSea))
    print(len(unique_vessels_monthly_SEAsia))
    #Plotting the vessel distribution
    timeperiod = list(range(0,len(unique_vessels_monthly_Atlantic)))
    fig, ax = plt.subplots()
    plt.xlabel ('Timeperiod')
    plt.ylabel('Vessel distribution over time')
    ax.plot(timeperiod,unique_vessels_monthly_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,unique_vessels_monthly_EastPacific,label = 'East Pacific')
    ax.plot(timeperiod,unique_vessels_monthly_WestPacific, label = 'West Pacific')
    ax.plot(timeperiod,unique_vessels_monthly_IndianOcean, label = 'Indian Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_Medittarnean, label = 'Mediettarnean Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_NorthSea,label = 'North Sea')
    ax.plot(timeperiod,unique_vessels_monthly_SEAsia, label = 'South East Asia')
    ax.plot(timeperiod,unique_vessels_monthly_Oceania, label = 'Oceania')
    legend = ax.legend(loc ='upper right',shadow = True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('small')
    for label in legend.get_lines():
        label.set_linewidth(1)
 #   plt.plot(x1,y1,x2,y2,x3,y3)
    plt.axis([0,len(timeperiod),0,100])
    plt.show()
    
    """
    fig, ax = plt.subplots()
    ax.plot(a, c, 'k--', label='Model length')
    ax.plot(a, d, 'k:', label='Data length')
    ax.plot(a, c+d, 'k', label='Total message length')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    # Set the fontsize
    for label in legend.get_texts():
    label.set_fontsize('large')

    for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
    plt.show()
    mintime = min(AtlanticTime)
    maxtime = max(AtlanticTime)
    deltatime = maxtime - mintime
    nom = LC.get_number_of_months(deltatime) #nom = number of months
    increment = deltatime/nom
    Timestamps = np.arange(mintime,maxtime,increment)
    number_of_messages_monthly_atlantic = [[] for i in range(0,len(Timestamps))]
    #Filter data to get it on a monthly basis 
    for i in range(0,len(AtlanticTime)):
        for j in range(1,len(Timestamps)):
            if j == 0:
                if AtlanticTime[i] < Timestamps[j+1]:
                    number_of_messages_monthly_atlantic[j].append(AtlanticTime[i])
 #           if j > 0 and j < len(Timestamps):        
 #               if AtlanticTime[i] > Timestamps[j-1] and AtlanticTime[i] < Timestamps[j]:
 #                   number_of_messages_monthly_atlantic[j].append(AtlanticMMSI[i])
            else:
                if AtlanticTime[i] > Timestamps[j-1] and AtlanticTime[i] < Timestamps[j]:
                    number_of_messages_monthly_atlantic[j].append(AtlanticMMSI[i])

    #The intervalltime will be given as an interval, hence there will be 11 elemnts for 12 months
    print(number_of_messages_monthly_atlantic)
    #number_of_messages_monthly_atlantic = number_of_messages_monthly_atlantic[:-1]
    unique_vessels_monthly_atlantic = [[] for i in range(0,len(number_of_messages_monthly_atlantic))]
    #Finding the unique number of messages per month     
    for i in range(0,len(number_of_messages_monthly_atlantic)):
        unique_vessels_monthly_atlantic[i] = len(set(number_of_messages_monthly_atlantic[i]))
    print(unique_vessels_monthly_atlantic)
    timeperiod = list(range(0,len(unique_vessels_monthly_atlantic)))
    print(timeperiod)
    """
    # Statistical Data
#   AtlanticMean = st.mean(AtlanticSpeed)
#    AtlanticVar = st.variance(AtlanticSpeed)
    """
    plt.figure()
    n, bins, patches = plt.hist([timeperiod,unique_vessels_monthly_atlantic])
    plt.xlabel('Timeperiod')
    plt.ylabel('Number of vessels')
    plt.title('Vessel Distribution of Time')
    x,y = [timeperiod,unique_vessels_monthly_atlantic]
    plt.plot(x,y)
    plt.axis([0,12,0,100])
    plt.show()
    Plotting  
    """
    """
    LC.plot_inside_polygons(AtlanticLon,AtlanticLat)       
    LC.plot_inside_polygons(EastPacificLon,EastPacificLat) 
    LC.plot_inside_polygons(WestPacificLon,WestPacificLat)
    LC.plot_inside_polygons(IndianOceanLon,IndianOceanLat)
    LC.plot_inside_polygons(MediterraneanLon,MediterraneanLat)
    LC.plot_inside_polygons(NorthSeaLon,NorthSeaLat)
    LC.plot_inside_polygons(SEAsiaLon,SEAsiaLat)
    LC.plot_inside_polygons(OceaniaLon,OceaniaLat)
    LC.plot_inside_polygons(LonOutsideZones,LatOutsideZones)
    """
    
####################################################
# THE FOLLOWING PART PLOTS HISTOGRAM OF SPEED DIST #
####################################################


from datetime import datetime
def GeoDistribution():
    global PacLat
    global PacLon
    global PacSpeed
    global AtlLat
    global AtlLon
    global AtlSpeed
    global SuezLat
    global SuezLon
    global SuezSpeed
    global PacMean
    global AtlMean
    global SuezMean
    global PacVol
    global AtlVol
    global SuezVol
    
    PacLat = list()
    PacLon = list()
    AtlLat = list()
    AtlLon = list()
    SuezLat = list()
    SuezLon = list()
    PacSpeed = list()
    AtlSpeed = list()
    SuezSpeed = list()
    
    for i in range(0,len(plotlat)):
        if plotlat[i] > 30 and (plotlon[i]>=150 or plotlon[i] <= -120):
            PacLat.append(plotlat[i])
            PacLon.append(plotlon[i])
            PacSpeed.append(speeds[i])
        elif plotlat[i]>10 and plotlon[i]>=-75 and plotlon[i]<= -10:
            AtlLat.append(plotlat[i])
            AtlLon.append(plotlon[i])
            AtlSpeed.append(speeds[i])
        elif plotlat[i] >= 0 and plotlat[i]<= 30 and plotlon[i] >=30 and plotlon[i] <= 100:
            SuezLat.append(plotlat[i])
            SuezLon.append(plotlon[i])
            SuezSpeed.append(speeds[i])
            
    PacMean = st.mean(PacSpeed)
    PacVol = st.stdev(PacSpeed)
    AtlMean = st.mean(AtlSpeed)
    AtlVol = st.stdev(AtlSpeed)
    SuezMean = st.mean(SuezSpeed)
    SuezVol = st.stdev(SuezSpeed)

def SpeedHistogram():
    plt.figure()
    hist, bins = np.histogram(speeds, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

    plt.figure()
    histval1, binsval = np.histogram(speeds, bins=50)
    histval = histval1/sum(histval1)
    width = 0.7 * (binsval[1] - binsval[0])
    center = (binsval[:-1] + binsval[1:]) / 2
    plt.bar(center, histval, align='center', width=width,color='k')
    plt.xlabel('Speed [knots]')
    plt.ylabel('Fraction of time')
    plt.show()

    
    plt.figure()
    histval1, binsval = np.histogram(draught, bins=20)
    histval = histval1/sum(histval1)
    width = 0.7 * (binsval[1] - binsval[0])
    center = (binsval[:-1] + binsval[1:]) / 2
    plt.bar(center, histval, align='center', width=width,color='k')
    plt.xlabel('Draught [meters]')
    plt.ylabel('Fraction of time')
    plt.show()
    
################################################
## THE FOLLOWING PARTS PLOTS POSITIONS ON MAPS #
################################################
def GlobalMap():
    plt.figure()
    lats = plotlat
    lons = plotlon
    m = Basemap(projection='hammer',lon_0=0)
    x, y = m(lons,lats)
    m.drawmapboundary(fill_color='#99ffff')
    #m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    m.scatter(x,y,10,marker='o',c='c',alpha=0.8)
    m.bluemarble()
    plt.show()

#########################################################
def LocalMap():
    '''
    clusterlat = list([  55.05639434, -118.20721601,  103.70615966,   32.42515   ,
        139.68368333,   39.15758613,   14.53560833,  148.2011    ,
         79.83414667,  121.63154333,   -5.43532222,    4.26149833,
         35.53425981,   54.03331389,  132.96183333,   23.508255  ,
       -122.31935778,  117.775     ,  101.26446286,   50.19599167,
         32.31820261])
        
    clusterlon = list([ 25.00425171,  33.7442034 ,   1.25787193,  30.26798929,
        35.40101667,  21.46019955,  35.82060833, -38.26788333,
         6.94608917,  31.34875667,  36.14576111,  51.35132167,
        33.90931056,  16.94192   ,  42.81025   ,  37.83974   ,
        37.79714722,  39.00233333,   2.8882519 ,  26.50235   ,  31.41053333])
        '''
    
    minlon = max(-180,min(plotlon)-5) #-10
    minlat = max(-90,min(plotlat)-5) #-10
    maxlon = min(180,max(plotlon)+5) #+10
    maxlat = min(90,max(plotlat)+5) #+10
    lat0 = (maxlat+minlat)/2
    lon0 = (maxlon+minlon)/2
    lat1 = (maxlat+minlat)/2-20

    fig=plt.figure(figsize=(18,18))
    fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='cyl',\
            lat_0=lat0,lon_0=lon0,lat_ts = lat1)

    m.drawmapboundary(fill_color='black')
    m.fillcontinents(color='dimgray',lake_color='black',zorder=0) #,zorder=0
    x, y = m(plotlon,plotlat) 
    #Ships:
    m.scatter(x,y,0.01,marker='o',c='yellow')
    #m.drawcoastlines()
    #Ports:
    #m.scatter(clusterlat,clusterlon,50,marker='x',c='magenta')
    
    m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])       
    m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
    #m.bluemarble()
    #plt.saveffig('/Users/PatrickAndreNaess/Desktop/PyPlots/current_plot.eps',\
    #            format='eps', dpi=1000)
   
################################################
## THE FOLLOWING PART PLOTS SPEED AGAINST TIME #
################################################

def SpeedDev(): 
    Timesteps = list()
    for i in range(0,len(timestep)):
        unixconv = datetime.fromtimestamp(timestep[i])
        Timesteps.append(unixconv)
 
    plt.figure()
    plt.plot(Timesteps,geospeed,'k')
    plt.show()
    plt.ylabel("Speed [knots]")   
    
def DraughtInterpolate():    
    global IntDraught    
    IntDraught = list()
    draughtSpeed = list()
    draughtIntTime = list()
    for i in range(0,len(timestep)):
        
        if timestep[i] >= draughttime[0]:
            idx = (np.abs(array(draughttime)-timestep[i])).argmin()
            IntDraught.append(draught[idx])
            draughtSpeed.append(speeds[i])
            draughtIntTime.append(timestep[i])
            print(i)
    
    speeds = 5
    draughts = 3
    operProf = np.zeros((draughts,speeds))
    mindraught = min(IntDraught)-0.01
    maxdraught = max(IntDraught)
    minspeed = 12
    maxspeed = max(draughtSpeed)
    deltaSpeed = maxspeed-minspeed
    deltaDraught = maxdraught-mindraught
    speedInt = deltaSpeed/speeds
    draughtInt = deltaDraught/draughts
    for i in range(0,len(IntDraught)):
        print(i)
        for a in range(0,draughts):
            for b in range(0,speeds):
                if draughtSpeed[i] > minspeed + b*speedInt and draughtSpeed[i] <= minspeed + (b+1)*speedInt and IntDraught[i] > mindraught + a*draughtInt and IntDraught[i] <= mindraught + (a+1)*draughtInt:
                    operProf[a,b] += 1
                    #print(a)
                    #print(b)
                
    
    sumOp = sum(operProf)
    normOp = np.zeros((draughts,speeds))               
    for a in range(0,draughts):
        for b in range(0,speeds):
            normOp[a,b] = operProf[a,b]/sumOp
                       
                    
def TradeDirectionSuez():
    global eastBoundlonS
    global westBoundlonS
    global eastBoundlatS
    global westBoundlatS
    global eastBoundSpeedS
    global westBoundSpeedS
    global eastBoundDraughtS
    global westBoundDraughtS
    global eastBoundTimeS
    global westBoundTimeS
    eastBoundlonS = list()
    westBoundlonS = list()
    eastBoundlatS = list()
    westBoundlatS = list()
    eastBoundSpeedS = list()
    westBoundSpeedS = list()
    eastBoundDraughtS = list()
    eastBoundTimeS = list()
    westBoundTimeS = list()
    
    
    for i in range(1,len(plotlat)):
        if plotlon[i] < 0 and plotlon[i] > - 11 and plotlat[i] > 40:
            if plotlat[i] < plotlat[i-1]:
                eastBoundlonS.append(plotlon[i])
                eastBoundlatS.append(plotlat[i])
                eastBoundSpeedS.append(speeds[i])
                eastBoundTimeS.append(timestep[i])
                #eastBoundDraught.append(dra)
            else:
                westBoundlonS.append(plotlon[i])
                westBoundlatS.append(plotlat[i])
                westBoundSpeedS.append(speeds[i])
                westBoundTimeS.append(timestep[i])
        elif plotlat[i] <= 40 and plotlon[i] < -7.5 and plotlon[i] >= -11 and plotlat[i] >= 35:
            
            if plotlat[i] < plotlat[i-1]:
                eastBoundlonS.append(plotlon[i])
                eastBoundlatS.append(plotlat[i])
                eastBoundSpeedS.append(speeds[i])
                eastBoundTimeS.append(timestep[i])
            else:
                westBoundlonS.append(plotlon[i])
                westBoundlatS.append(plotlat[i])
                westBoundSpeedS.append(speeds[i])
                westBoundTimeS.append(timestep[i])
        elif plotlon[i] >= -7.5 and plotlat[i] < 45 and plotlat[i] >= 0 and plotlon[i] <= 110:
            if plotlon[i] > plotlon[i-1]:
                eastBoundlonS.append(plotlon[i])
                eastBoundlatS.append(plotlat[i])
                eastBoundSpeedS.append(speeds[i])
                eastBoundTimeS.append(timestep[i])
            else:
                westBoundlonS.append(plotlon[i])
                westBoundlatS.append(plotlat[i])
                westBoundSpeedS.append(speeds[i])
                westBoundTimeS.append(timestep[i])
                

    minlon = max(-180,min(westBoundlonS)-10)
    minlat = max(-90,min(westBoundlatS)-10)
    maxlon = min(180,max (westBoundlonS)+10)
    maxlat = min(90,max(westBoundlatS)+10)
    lat0 = (maxlat+minlat)/2
    lon0 = (maxlon+minlon)/2
    lat1 = (maxlat+minlat)/2-20

    fig=plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='cyl',\
            lat_0=lat0,lon_0=lon0,lat_ts = lat1)

    m.drawcoastlines()
    #m.fillcontinents()
    x, y = m(eastBoundlonS,eastBoundlatS)
    X, Y = m(westBoundlonS,westBoundlatS)
    m.scatter(x,y,10,marker='o',c='k',alpha = 1)
    m.scatter(X,Y,10,marker='D',c='w',alpha = 0.1)
    m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])       
    m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
    m.bluemarble()
#    #m.bluemarble()

    plt.figure()
    bins = np.linspace(0, 25, 51)
    plt.hist(westBoundSpeedS, bins, alpha=0.6, label='West Bound',color='k',normed=True)
    plt.hist(eastBoundSpeedS, bins, alpha=0.6, label='East Bound',color='w',normed=True)
    plt.legend(loc='upper right')
    plt.xlabel('Speed [knots]')
    plt.ylabel('Fraction of time')
    plt.show()
                
def FreightRateImpact():
    global highRateSpeed
    global medRateSpeed
    global lowRateSpeed
    
    highRateSpeed = list()
    medRateSpeed = list()
    lowRateSpeed = list()
    
    for i in range(0,len(speeds)):
        if timestep[i] > 1293840000 and timestep[i] <= 1309392000:
            highRateSpeed.append(speeds[i])
        elif timestep[i] > 1388534400 and timestep[i] <= 1404086400:
            lowRateSpeed.append(speeds[i])
        elif timestep[i] > 1422748800 and timestep[i] <= 1438300800:
            medRateSpeed.append(speeds[i])
    
    
def TradeDirectionPacific():
    global eastBoundlonP
    global westBoundlonP
    global eastBoundlatP
    global westBoundlatP
    global eastBoundSpeedP
    global westBoundSpeedP
    eastBoundlonP = list()
    westBoundlonP = list()
    eastBoundlatP = list()
    westBoundlatP = list()
    eastBoundSpeedP = list()
    westBoundSpeedP = list()
    
    for i in range(1,len(plotlat)):
        if (plotlon[i] > 140 or plotlon[i] < -130) and plotlat[i] > 30:
            if plotlon[i] > plotlon[i-1]:
                eastBoundlonP.append(plotlon[i])
                eastBoundlatP.append(plotlat[i])
                eastBoundSpeedP.append(speeds[i])
            elif plotlon[i] < -160 and plotlon[i-1] > 170:
                eastBoundlonP.append(plotlon[i])
                eastBoundlatP.append(plotlat[i])
                eastBoundSpeedP.append(speeds[i])
            elif plotlon[i] < plotlon[i-1]:
                westBoundlonP.append(plotlon[i])
                westBoundlatP.append(plotlat[i])
                westBoundSpeedP.append(speeds[i])
            elif plotlon[i] > -160 and plotlon[i-1] < 170:
                westBoundlonP.append(plotlon[i])
                westBoundlatP.append(plotlat[i])
                westBoundSpeedP.append(speeds[i])

    pacificSpeeds = westBoundSpeedP+eastBoundSpeedP
    std = st.stdev(pacificSpeeds)
    mean = st.mean(pacificSpeeds)
    plt.figure()
    bins = np.linspace(0, 25, 51)
    plt.hist(westBoundSpeedP+eastBoundSpeedP, bins, alpha=0.6, label='December',color='k',normed=True)
    plt.legend(loc='upper right')
    plt.xlabel('Speed [knots]')
    plt.ylabel('Fraction of time')
    plt.title('Speed mean = %s, Speed std = %s'%(mean,std))
    plt.show()       

def MonthlyAverage():
    global MarketRates
    MarketRates = pandas.read_excel('/Users/jonleonhardsen/Documents/Documents/Skole/NTNU vår 2017/RatesOilPrice.xlsx')
    global monthlyMeanWS
    global monthlyMeanES
    global monthlyMean
    global monthlyVol
    global months
    global freightRates
    global oilPrice
    global sortedtimeWS
    global sortedtimeES
    global countMess
    global monthsCon
    august2010 = 1280620800
    december2015 = 1451606354
    Intervals = 5+5*12
    Increments = round((december2015-august2010)/Intervals)
    monthlyMeanWS = list()
    monthlyMeanES = list()
    monthlyMean = list()
    monthlyVol = list()
    months = list()
    freightRates = list()
    oilPrice = list()
    countMess = list()
    marketData = MarketRates.as_matrix()
    sortedtimeWS, sortedspeedWS = zip(*sorted(zip(westBoundTimeS, westBoundSpeedS)))
    sortedtimeES, sortedspeedES = zip(*sorted(zip(eastBoundTimeS, eastBoundSpeedS)))
    sortedtime, sortedspeed = zip(*sorted(zip(timestep, geospeed)))
    
    count = 1
    countW = 1
    countE = 1
    for i in range(0,Intervals-1):
        print(i)
        monthlyspeedWS = list()
        monthlyspeedES = list()
        monthlyspeed = list()

        j = countW         
        while sortedtimeWS[j] <= august2010 + Increments*i and sortedtimeWS[j] > august2010 + Increments*(i-1):
                monthlyspeedWS.append(sortedspeedWS[j])
                j += 1
        countW = j+1
        
        if len(monthlyspeedWS)>1:        
            monthlyMeanWS.append(st.mean(monthlyspeedWS))
        else:
            monthlyMeanWS.append(0)
       
        j = countE    
        while sortedtimeES[j] <= august2010 + Increments*i and sortedtimeES[j] > august2010 + Increments*(i-1):
                monthlyspeedES.append(sortedspeedES[j])
                j += 1
        countE = j+1
        
        if len(monthlyspeedES)>1:
            monthlyMeanES.append(st.mean(monthlyspeedES))
        else:
            monthlyMeanES.append(0)
        
        j = count   
        while sortedtime[j] <= august2010 + Increments*i and sortedtime[j] > august2010 + Increments*(i-1):
                monthlyspeed.append(sortedspeed[j])
                j += 1
        count = j+1
        
        if len(monthlyspeed)>1:
            monthlyMean.append(st.mean(monthlyspeed))
        else:
            monthlyMean.append(0)
        
        countMess.append(len(monthlyspeed))
        months.append(august2010+Increments*i)
        monthsCon = list()
        for i in range(0,len(months)):
            unixconv = datetime.fromtimestamp(months[i])
            monthsCon.append(unixconv)
        
        freightRates.append(marketData[i][1])
        oilPrice.append(marketData[i][2])
        
    plt.figure()
    plt.bar(monthsCon,monthlyMeanWS,alpha = 0.8, color='k',label='West Bound Suez')
    plt.bar(monthsCon,monthlyMeanES,alpha = 0.8, color='w',label='East Bound Suez')
    plt.xlabel('Months')
    plt.ylabel('Speed [knots]')
    plt.legend(loc='upper right')
    
    plt.figure()
    plt.bar(monthsCon,monthlyMean,alpha = 0.8, color='k',label='Global')
    
    plt.xlabel('Months')
    plt.ylabel('Speed [knots]')
    plt.legend(loc='upper right')
    
    
    years = [1293796800.0, 1325332800.0, 1356868800.0, 1388404800.0, 1419940800.0, 1451476800.0]
    yearlyAv = list()
    yearSpeed = list()
    for i in range(1,len(years)):
        yearSpeed = list()
        for j in range(0,len(monthlyMean)):
            if months[j] <= years[i]:
                yearSpeed.append(monthlyMean[j])
        
        yearlyAv.append(st.mean(yearSpeed))
    
    yearsCon = list()
    del years[-1]
    for i in range(0,len(years)):
        unixconv = datetime.fromtimestamp(years[i])
        yearsCon.append(unixconv)
   


############################################################################
## THE FOLLOWING PART CALCULATES INTERVALS OF MESSAGES AND PLOTS HISTOGRAM #
############################################################################
def SpeedForAnalysis():
    global logdiff
    global UpDiff
    global UpTime
    global UpSpeed
    global AnSpeed
    global AnTime
    global AnSteps
    global AnDiff
    global AnLat
    global AnLon
    logdiff = list()
    logdiff = np.diff(timestep)  
    UpDiff = list()
    UpSpeed = speeds
    UpTime = timestep
    UpDiff = logdiff
    UpLat = plotlat
    UpLon = plotlon
    
    for i in range(0,len(logdiff)):
        if logdiff[i] < 1:
            logdiff[i] = logdiff[i]
            UpSpeed[i+1] = 101010
            UpTime[i+1] = 101010
            UpDiff[i] = 101010

    AnSpeed = [x for x in UpSpeed if x != 101010] #AnDiff = Message intervals for analysis
    AnTime = [x for x in UpTime if x != 101010]
    AnLat = [x for x in UpLat if x != 101010]
    AnLon = [x for x in UpLon if x != 101010] #AnSpeed = Speed for analysis
    AnDiff = [x for x in UpDiff if x != 101010] #AnTime = Time for analysis
    AnDiff.append(1)
    AnSteps = list()
    for i in range(0,len(AnTime)):
        Unixconv = datetime.fromtimestamp(AnTime[i])
        AnSteps.append(Unixconv)


################################################
## THE FOLLOWING PART PLOTS INTERVAL FREQUENCY #
################################################
def IntervalFreq():    
    Count = list()
    Number = list()
    for i in range(0,360):
        Count.append(i)
        Number.append(AnDiff.count(i))
    
    plt.figure()
    plt.plot(Count,Number)
    plt.show()   
    plt.figure()
    plt.scatter(AnSteps,AnSpeed,c='k')
    #plt.plot(IncSteps,AvSpeed,'k')
    plt.show()
    
    
def GeoAnalysis(max_lon,max_lat,min_lon,min_lat,minspeed):
    global AtSpeed
    global AtTime
    global AtLat
    global AtLon
    global AtSteps
    
    AtMaxLon = max_lon
    AtMaxLat = max_lat
    AtMinLon = min_lon
    AtMinLat = min_lat
    AtSpeed = list()
    AtTime = list()
    AtLat = list()
    AtLon = list()

    for i in range(0,len(AnSpeed)):
        #print(i)
        if AnSpeed[i] > minspeed and AnLat[i] > AtMinLat and AnLat[i] < AtMaxLat and (AnLon[i] > AtMinLon or AnLon[i] < AtMaxLon):
            AtSpeed.append(AnSpeed[i])
            AtTime.append(AnTime[i])
            AtLat.append(AnLat[i])
            AtLon.append(AnLon[i])
        
    AtSteps = list()
    for i in range(0,len(AtTime)):
        Unixconv = datetime.fromtimestamp(AtTime[i])
        AtSteps.append(Unixconv)


def DraughtAnalysis():    
    global AnDraught
    global AnCat
    global DSpeed
    global DTime    
    global Dcat
    global slope
    global intercept
    global r_value
    global p_value
    global std_err
    Drange = max(draught)-min(draught)
    Intervals = 3
    Dsteps = Drange/Intervals
    Dcat = [0]*len(draught)

    for i in range(0,len(draught)):
        for j in range(0,Intervals-1):
            if  min(draught) + Dsteps*j <= draught[i] <= min(draught)+(j+1)*Dsteps :
                Dcat[i] = j+1

    AnDraught = list()
    AnCat = list() 
    DSpeed = list()
    DTime = list()
    for j in range(0,len(Dcat)-1):
        for i in range(0,len(AnTime)):
        #if AnTime[i]
            if AnTime[i]>=draughttime[j] and AnTime[i]<draughttime[j+1]:
                AnDraught.append(draught[j])
                AnCat.append(Dcat[j])
                DSpeed.append(AnSpeed[i])
                DTime.append(AnTime[i])
    slope, intercept, r_value, p_value, std_err = stats.linregress(AnDraught,DSpeed) 

    
def CumulatedSpeed():
    import math
    global SRange
    global SortSpeed
    global SpeedDist
    global CumSpeed
    global NSpeedDist
    global NCumSpeed
    global InterSpeed
    SRange = math.ceil(max(AnSpeed))
    SortSpeed = np.sort(AnSpeed)
    SpeedDist = [0]*SRange
    CumSpeed = [0]*SRange
    InterSpeed = [0]*SRange
    for i in range(0,len(AnSpeed)):
        for j in range(1,SRange+1):
            InterSpeed[j-1] = j
            if SortSpeed[i] < j:
                CumSpeed[j-1] = CumSpeed[j-1] + 1
                if SortSpeed[i] >= j-1:
                    SpeedDist[j-1] = SpeedDist[j-1] + 1

    NSpeedDist = [0]*SRange
    NCumSpeed = [0]*SRange
    for i in range(0,SRange):
        NSpeedDist[i] = SpeedDist[i]/len(AnSpeed)
        NCumSpeed[i] = CumSpeed[i]/len(AnSpeed)
        

if __name__ == "__main__":       
    ExtractData(filepath,a,b,c,d)#'/Volumes/LaCie/SAVONA.db')
    GlobalMap()
    LocalMap()
    SpeedDev()
    SpeedHistogram()
    SpeedForAnalysis()
    IntervalFreq()
    GeoAnalysis()
    DraughtAnalysis()
    CumulatedSpeed()

