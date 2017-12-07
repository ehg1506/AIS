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
from datetime import datetime
import statistics as st

      
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
    polygon_Eastpacific =[[-142,70] ,
                      [-103.227539,25.443275] ,
                      [-76.245117,7.31882],
                      [-60.644531,-18.646245] ,
                      [-73.476562,-62.267923] ,
                      [-180,-36.315125],
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

def get_timevector(lt,ht):
    deltatime = ht - lt 
    mintime = datetime.fromtimestamp(lt)
    maxtime = datetime.fromtimestamp(ht)
       
    months= (maxtime.year - mintime.year)*12 + maxtime.month - mintime.month
    
    increment = deltatime/months
    Timestamp = list()
    Timestamp.append(lt)
    for i in range(1,months):
        Timestamp.append(Timestamp[i-1]+increment)
    
    Timestamp.append(ht)
    
    return Timestamp

def monthly_filter(DF,timestamps):
    monthly_mean_speeds = [[] for i in range(0,(len(timestamps)-1))]
    monthly_unique = [[] for i in range(0,(len(timestamps)-1))]

    for j in range(0,len(monthly_mean_speeds)):
        X = DF[(DF['Unixtime'] >= timestamps[j]) & (DF['Unixtime'] < timestamps[j+1])]
        
        if len(X['Speed']) == 0:
            monthly_mean_speeds[j].append(0)
        else:
            monthly_mean_speeds[j].append(st.mean(X['Speed'].tolist()))
        monthly_unique[j].append(X['MMSI'].nunique())
                    
    return monthly_mean_speeds,monthly_unique

def centeroidnp(arr):
    x = [p[0] for p in arr]
    y = [p[1] for p in arr]
    return (sum(x) / len(arr), sum(y) / len(arr))
