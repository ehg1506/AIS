#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:07:07 2017

@author: PatrickAndreNaess
"""


import folium

m = folium.Map(location=[0, 0], attr='XXX Mapbox Attribution',\
               tiles="cartodbdark_matter",\
               zoom_start=2)

lats = BigD[BigD['MMSI']==211549000]['Lat']
lons = BigD[BigD['MMSI']==211549000]['Lon'] 

for i in range(len(lats)):
    folium.CircleMarker([lats[i], lons[i]],radius=0.1,color = '#81c487').add_to(m)

folium.PolyLine(list(zip(lats, lons)),color = '#81c487',opacity = 0.2).add_to(m)

m.save('/Users/PatrickAndreNaess/Desktop/#NYINTERAKTIVplot.html')