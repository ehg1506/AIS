#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:12:39 2017

@author: erikgrundt
"""
#Location is a number from 1 to 5, corresponding to the zone of interrest
#2 and 3 is the pacific

import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sqlite3
from mpl_toolkits.basemap import Basemap
import datetime

      
def generate_polygons():
    global polygons
    polygons = list()
    # Atlantic
    polygon_atlantic = [[31.464844,-44.840291], 
                        [25,-10],
                        [-5.493164,35.942436],
                        [11.118164,54.110934],  
                        [-37.265,65.219],
                        [-100,50],
                        [-103.227539,25.443275],
                        [-76.245117,7.31882],
                        [-60.644531,-18.646245],
                        [-73.476562,-62.267923]] 
    polygon_atlantic.append(polygon_atlantic[0]) #Adding the first point 
    polygons.append(polygon_atlantic)
    # East Pacific 
    polygon_Eastpacific =[[-142,60] ,
                        [-103.227539,25.443275] ,
                        [-76.245117,7.31882],
                        [-60.644531,-18.646245] ,
                        [-73.476562,-62.267923] ,
                        [-179.296875,-36.315125],
                        [-180,47.989922],
                        [-180,58.077876]]
    polygon_Eastpacific.append(polygon_Eastpacific[0])
    polygons.append(polygon_Eastpacific)
    # West Pacific
    polygon_WestPacific = [[180,60],
                [143.525391,59.085739],
                [118.828125,38.959409],
                [105.292969,21.453069],
                [121.816406,17.978733],
                [131.835938,1.230374],
                [180,-30]]
    polygon_WestPacific.append(polygon_WestPacific[0])
    polygons.append(polygon_WestPacific)
    #Indian Ocean
    polygon_indi = [[33.04687,29.075375],
                    [80,40],
                    [76.992188,11.523088], 
                    [96.503906,-14.093957],
                    [85.078125,-57.515823],
                    [31.464844,-44.840291],
                    [25,-10]]
    polygon_indi.append(polygon_indi[0])
    polygons.append(polygon_indi)
    #Mediterranean Ocean
    polygon_medi = [[-5.097,35.960], 
                    [11.118164,54.110934],
                    [50,50],
                    [33.04687,29.075375],
                    [11.777,33.358]]
    polygon_medi.append(polygon_medi[0])
    polygons.append(polygon_medi)

    #North Sea 
    polygon_northSea = [[11.118164,54.110934],  
                        [50,50],
                        [16.699219,76.351896],
                        [-19.160156,76.184995],
                        [-37.265,65.219]]
    polygon_northSea.append(polygon_northSea[0])
    polygons.append(polygon_northSea)
    #South East Asia
    polygon_EA = [[105.292969,21.453069],
                  [121.816406,17.978733],
                  [131.835938,1.230374],
                  [126.210938,-10.487812],
                  [96.503906,-14.093957],
                  [76.992188,11.523088],
                  [80,40]]
    polygon_EA.append(polygon_EA[0])
    polygons.append(polygon_EA)    
    #Oceania
    polygon_O = [[131.835938,1.230374],
                 [180,-30],
                 [180,-50],
                 [131.132813,-57.231503],
                 [85.078125,-57.515823],
                 [96.503906,-14.093957],
                 [126.210938,-10.487812],
                 [131.835938,1.230374]]
    polygon_O.append(polygon_O[0])
    polygons.append(polygon_O)   
    
    '''
    #Small zone aoround
    #Suez and Red Sea
    #Can be included in further analysis
    PointS_1 = [31.591187,30.656816]
    PointS_2 = [33.063354,30.883369]
    PointS_3 = [33.189697,27.907058]
    PointS_4 = [33.892822,28.15919]
    PointS_5 = [38.320313,14.264383]
    PointS_6 = [44.121094,17.308688]
    polygon_suez = [PointS_1, PointS_2, PointS_3, PointS_4, PointS_5, PointS_6]
    polygon_suez.append(polygon_suez[0])
    polygons.append(polygon_suez)
    '''
    return polygons


def point_inside_polygon(x,y,poly):
    
    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


#Takes an input argument of polygons and plotting them! Should be 
def ocean_polygon(polygons):
    plt.figure(figsize=(20,20))
    m = Basemap(projection='cyl',lon_0=0,resolution='l')
    m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1],color='k')       
    m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1],color='k')
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='black',lake_color='white')
    m.drawcountries()
    m.drawcoastlines()
    
    for i in range(0,len(polygons)):
        x,y = zip(*polygons[i])
        m.plot(x,y,marker='.')
        
    plt.show()    
    #plt.savefig('/Users/erikgrundt/Desktop/currentpoly.eps',\

def plot_inside_polygons(lon,lat):
    plt.figure(figsize=(20,20))
    m = Basemap(projection='cyl',lon_0=0,resolution='l')
    m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1],color='k')       
    m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1],color='k')
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='black',lake_color='white')
    m.drawcountries()
    m.drawcoastlines()
    x,y = m(lon,lat)
    m.scatter(x,y,0.01,marker='o',c='blue')
    plt.show()
    
def get_number_of_months(u):
    d = datetime.datetime.fromtimestamp(u)
    return d.month

def find_unique_vessels(vessellist,unique_vessels):
    for i in range(0,len(vessellist)):
        if vessellist[i] not in unique_vessels:
            unique_vessels.append(vessellist[i])
    return unique_vessels

def get_timevector(Timelist):
    mintime = min(Timelist)
    maxtime = max(Timelist)
    deltatime = maxtime - mintime
    d = datetime.datetime.fromtimestamp(deltatime)
    months_from_years = 12*(d.year - 1970)
    months_left = d.month
    months = months_left + months_from_years
    increment = deltatime/months
    Timestamp = np.arange(mintime,maxtime,increment)
    return Timestamp

def get_generic_timevector(mintime,maxtime):
    deltatime = maxtime - mintime
    d = datetime.datetime.fromtimestamp(deltatime)
    months_from_years = 12*(d.year - 1970)
    months_left = d.month
    months = months_left + months_from_years
    increment = deltatime/months
    Timestamp = np.arange(mintime,maxtime,increment)
    return print(Timestamp)

def monthly_filter(timelist,mmsi,timestamps,monthly_list):
    for i in range(0,len(timelist)):
        for j in range(1,len(timestamps)):
            if j== 0:
                if timelist[i] < timestamps[j+1]:
                    monthly_list[j].append(mmsi[i])
            else:
                if timelist[i] > timestamps[j-1] and timelist[i] < timestamps[j]:
                    monthly_list[j].append(mmsi[i])
    return monthly_list
    
def unique_vessels_monthly(messages_monthly,unique_vessel_monthly):
    for i in range(0,len(messages_monthly)):
        if messages_monthly[i] not in unique_vessel_monthly[i]:
            unique_vessel_monthly[i] = len(set(messages_monthly[i]))
    return unique_vessel_monthly



#minmaxtime = (1325376000,1420070400)

#timelist2 = get_timevector(timelist1)
#print(timelist2)

"""            
    
Timestamps = np.arange(mintime,maxtime,increment)
    
    
    
for i in range(0,len(number_of_messages_monthly_atlantic)):
    unique_vessels_monthly_atlantic[i] = len(set(number_of_messages_monthly_atlantic[i]))
print(unique_vessels_monthly_atlantic)
    
    
    number_of_messages_monthly_atlantic = [[] for i in range(0,len(Timestamps))]
    #Filter data to get it on a monthly basis 
    for i in range(0,len(AtlanticTime)):
        for j in range(0,len(Timestamps)):
            if j == 0:
                if AtlanticTime[i] < Timestamps[j+1]:
                    number_of_messages_monthly_atlantic[j].append(AtlanticTime[i])
            if j > 0 and j < len(Timestamps):        
                if AtlanticTime[i] > Timestamps[j-1] and AtlanticTime[i] < Timestamps[j]:
                    number_of_messages_monthly_atlantic[j].append(AtlanticMMSI[i])
            else:
                if AtlanticTime[i] > Timestamps[j-1] and AtlanticTime[i] < Timestamps[j]:
                    number_of_messages_monthly_atlantic[j].append(AtlanticMMSI[i])

    #The intervalltime will be given as an interval, hence there will be 11 elemnts for 12 months
    number_of_messages_monthly_atlantic = number_of_messages_monthly_atlantic[:-1]

"""    
"""
    #Make on list containing the number of unqiue vessel found in a time interval of one month
    mintime = min(AtlanticTime)
    maxtime = max(AtlanticTime)
    deltatime = maxtime - mintime
    nom = LC.get_number_of_months(deltatime) #nom = number of months
    increment = deltatime/nom
    
    #List for timestamps
    Timestamps = np.arange(mintime,maxtime,increment)
"""
          
#Running
#extract_geodata(1,'/Users/erikgrundt/Desktop/ContainerFleet.db') 
#generate_polygons()
#ocean_polygon(polygons)
#print(point_inside_polygon(0,0,polygons[0]))