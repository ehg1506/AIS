#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:07:07 2017

@author: PatrickAndreNaess
"""


import folium
import numpy as np
from folium.plugins import FastMarkerCluster


lats = BigD['Lat']
lons = BigD['Lon']     
       
       
m = folium.Map(location=[0,0], attr='XXX Mapbox Attribution',\
               tiles="cartodbdark_matter",\
               zoom_start=2)

FastMarkerCluster(data=list(zip(lats, lons))).add_to(m)

folium.LayerControl().add_to(m)

m.save('/Users/PatrickAndreNaess/Desktop/PyPlots/NYINTERAKTIVmeld.html')