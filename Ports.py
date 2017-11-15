#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:42:19 2017

@author: PatrickAndreNaess
"""

import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

file = '/Users/erikgrundt/Documents/Phyton/ports.csv'

portdata = pd.DataFrame(pd.read_csv(file,sep="\t"))

# Filter NaNs from lat and lon, and D_region or Country of intrest:
# US West Coast, West Med, North Continent Europe 

port_lon = portdata['Longitude'].dropna()
port_lat = portdata['Latitude'].dropna()
#port_lon = portdata[(portdata['Country']=='Italy')]['Longitude'].dropna()
#port_lat = portdata[(portdata['Country']=='Italy')]['Latitude'].dropna()

def PortMap(lo,la):
    minlon = max(-180,min(port_lon)-5) #-10
    minlat = max(-90,min(port_lat)-5) #-10
    maxlon = min(180,max(port_lon)+5) #+10
    maxlat = min(90,max(port_lat)+5) #+10
    
    plt.figure(figsize=(10,10))
    m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,\
                urcrnrlat=maxlat,resolution='i',projection='cyl',lat_0=0)

    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='black',lake_color='white')
    m.drawcountries()
    m.drawcoastlines()
    #m.fillcontinents(color='gray',zorder=0)
    x, y = m(lo,la)
    m.scatter(x,y,10,marker='o',c='blue')
    m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1],color='k')       
    m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1],color='k')

PortMap(port_lon,port_lat)

#Number of ports per country:
land = portdata.groupby('Country').count()['name'].sort_values(ascending=False)
print(land)

#Number of ports per region:
land = portdata.groupby('D_Region').count()['name'].sort_values(ascending=False)
print(land)

'''
#Filter to analyse region:
region = portdata[(portdata['Longitude']>40) & (portdata['Longitude']<60) & \
         (portdata['Latitude'] > -10) & (portdata['Latitude'] < 10)]
print(region)
'''