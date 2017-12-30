#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:12:39 2017

@author: erikgrundt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import statistics as st
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import pandas as pd

      
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
                      [-100,50] ,
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
                [130,62],
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
                    [25,-10]]
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
    fig, ax = plt.subplots(figsize=(20,20))
    m = Basemap(projection='cyl',lon_0=0,resolution='l')
    #m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1],color='k')       
    #m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1],color='k')
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgrey',lake_color='white')
    #m.drawcountries()
    #m.drawcoastlines()
    
    for i in range(0,len(polygons)):
        x,y = zip(*polygons[i])
        m.plot(x,y,marker='.')
        ax.fill(x, y,alpha=0.2)
    plt.show()    
    #plt.savefig('/Users/erikgrundt/Desktop/currentpoly.eps',\
    
    polygons = generate_polygons()
    ocean_polygon(polygons)


def plot_inside_polygons(lon,lat):
    fig, ax = plt.subplots(figsize=(18,18))
    m = Basemap(projection='cyl',lon_0=0,resolution='l')
    #m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1],color='k')       
    #m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1],color='k')
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgray',lake_color='white')
    x,y = m(lon,lat)
    m.scatter(x,y,0.01,marker='o',c='black')
        
    x,y = zip(*polygons[0])
    m.plot(x,y,marker='.')
    ax.fill(x, y,alpha=0.2)

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
    monthly_message_interval = [[] for i in range(0,(len(timestamps)-1))]
    monthly_stdev_speeds = [[] for i in range(0,(len(timestamps)-1))]

    for j in range(0,len(monthly_mean_speeds)):
        X = DF[(DF['Unixtime'] >= timestamps[j]) & (DF['Unixtime'] < timestamps[j+1])]
        
        #SPEED
        if len(X['Speed']) == 0:
            monthly_mean_speeds[j].append(0)
            monthly_stdev_speeds[j].append(0)
        else:
            monthly_mean_speeds[j].append(st.mean(X['Speed']))
            monthly_stdev_speeds[j].append((X['Speed']))#st.stdev
        
        #MESSAGE INTERVAL
        if len(X['Unixtime']) < 2:
            hours = 24*30 # One month if no messages, in hours
        else:    
            time = st.mean(abs(np.diff(X['Unixtime'])))
            hours = (datetime.fromtimestamp(time)-datetime(1970,1,1)).total_seconds()/60/60
        monthly_message_interval[j].append(hours)
        
        #UNIQUE VESSELS
        monthly_unique[j].append(X['MMSI'].nunique())
                    
    return monthly_mean_speeds,monthly_stdev_speeds,monthly_unique,monthly_message_interval

def percentageMonthly(zone, world):
    percent = list()
    
    for i in range(0,len(world)):
        p = zone[i][0]/world[i][0]*100
        percent.append(p)
        
    return percent

def get_centermost_point(plotlon,plotlat,n_clusters_,labels):
    df = pd.DataFrame({'lon': plotlon, 'lat': plotlat})
    coords = df.as_matrix(columns=['lon', 'lat'])
    clusters = pd.Series([coords[labels==n] for n in range(n_clusters_)])
    
    centroid = list()
    for i in range(0,len(clusters)):
        if len(clusters[i]) >=1:
            centroid.append((MultiPoint(clusters[i]).centroid.x, MultiPoint(clusters[i]).centroid.y))
    
    centermost_point = list()
    for i in range(0,len(clusters)):
        if len(clusters[i]) >=1:
            centermost_point.append(min(clusters[i], key=lambda point: great_circle(point, centroid[i]).m))

    return tuple(centermost_point)
