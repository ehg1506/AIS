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
import pandas as pd
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as hcluster
import loc_check as LC
import networkx as nx
from itertools import cycle, groupby


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
    
    draughttime = list()
    draught = list()
    useridDraught = list()

    conn = sqlite3.connect(filepath)
    cur = conn.cursor()
    
    # For nav-igation status, insert in '': 
    portstatus = 'and (nav_status==1 or nav_status ==5)'
    
    SQLstring1 = "SELECT unixtime,sog,latitude,longitude,userid,\
        nav_status FROM %s WHERE longitude<= %s and latitude <= %s\
        and longitude >= %s and latitude >= %s and sog >= %s and \
        sog <= %s and unixtime >= %s and unixtime <= %s %s ORDER BY UNIXTIME \
        ASC" % ('Panamax1',str(a),str(b),str(c),str(d),str(minspeed),\
        str(maxspeed),str(lowtime),str(hightime),portstatus)
    
    A = time.time()
    with conn:
        cur = conn.cursor()  
        if SegmentAnalysis == 1:
            cur.execute(SQLstring1)
        else:
            cur.execute("SELECT unixtime,sog,latitude,longitude,userid FROM MessageType1 ORDER BY UNIXTIME ASC")
        VesselData = cur.fetchall()
        
        for i in range(0,len(VesselData)):
            Datastrip = VesselData[i]
            timestep.append(Datastrip[0])
            speeds.append(Datastrip[1])
            plotlat.append(Datastrip[2])
            plotlon.append(Datastrip[3])        
            mmsi.append(Datastrip[4])
            navstat.append(Datastrip[5])      
    
    cur.close()
    
    conn = sqlite3.connect(filepath)
    cur = conn.cursor()
    
    SQLstring2 = "SELECT unixtime,draught,userid from %s where \
        unixtime >= %s and unixtime <= %s" % ('BigShips5',str(lowtime),\
        str(hightime))
    
    with conn:
        cur = conn.cursor()
        if SegmentAnalysis == 1:
            cur.execute(SQLstring2)
        else:
            cur.execute("SELECT unixtime,draught,userid FROM MessageType5 ORDER BY UNIXTIME ASC")
        draughtdata = cur.fetchall()
         
    for i in range(0,len(draughtdata)):
        draughtstrip = draughtdata[i]
        if draughtstrip[1]/10 > 5:
            draughttime.append(draughtstrip[0])
            draught.append(draughtstrip[1]/10)
            useridDraught.append(draughtstrip[2])
            
    print('Database extraction time is: ' + str(time.time()-A) + ' s')
    
    return plotlon, plotlat
'''
def DataFrameForAnalysis():
    # Lager df for analyse            
    df = pandas.DataFrame({'Speed': speeds, 'MMSI': mmsi, 'Unixtime': timestep, \
                             'Lat': plotlat, 'Lon': plotlon, 'Nav status': navstat})
    df['Date/Time'] = df['Unixtime'].apply(lambda x: \
                      datetime.fromtimestamp(x).strftime("%d/%m/%Y %H:%M:%S"))
    return df
'''

def PortsPlot(lon,lat,clon,clat,algorithm):
    minlon = max(-180,min(lon)-5) #-10
    minlat = max(-90,min(lat)-5) #-10
    maxlon = min(180,max(lon)+5) #+10
    maxlat = min(90,max(lat)+5) #+10
    lat0 = (maxlat+minlat)/2
    lon0 = (maxlon+minlon)/2
    lat1 = (maxlat+minlat)/2-20

    fig = plt.figure(figsize=(18,18))
    fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='cyl',\
            lat_0=lat0,lon_0=lon0,lat_ts = lat1)

    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgray',lake_color='white',zorder=0) #,zorder=0
    x, y = m(lon,lat) 
    #Ships:
    ships = m.scatter(x,y,1,marker='o',c='black', label = 'AIS data')
    
    #Ports:
    ports = m.scatter(clon,clat,50,marker='x',c='red', alpha = 0.5,\
                      label = 'Estimated ports')
    
    plt.legend(handles=[ships,ports],prop={'size': 12})
    #plt.title('AIS data vs. Estimated port locations with %s' % algorithm)
    
    #m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])       
    #m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
    plt.savefig('/Users/PatrickAndreNaess/Desktop/PyPlots/current_plot.eps',\
                format='eps', dpi=1000)

def ClusterPorts():
    global clusterlat
    global clusterlon
    global n_clusters_
    
    points = np.stack((plotlon,plotlat),axis=-1)
    N = len(points)
    
    # 1 = Hierarchical clustering algorithm
    # 2 = K-means clustering based on number of cluster found with # 1
    # 3 = Affinity propagation clustering algorithm
    # 4 = Mean-shift clustering algorithm
    # 5 = DBSCAN clustering algorithm
    
    CLUSTER = 5
    
    if (CLUSTER == 1) or (CLUSTER == 2):
        #Cluster with a Hierarchical clustering algorithm to get k:
        thresh = 2
        labelsHiarch = hcluster.fclusterdata(points,thresh,criterion="distance")
        
        #Get the number of clusters
        n_clusters_ = pd.Series(labelsHiarch).nunique()
        
        print('Estimated number of clusters: '+str(n_clusters_)+' ports, from '+\
              str(N)+' messages.')     
        
        # calculate full dendrogram
        plt.figure(figsize=(18,6))
    
        # generate the linkage matrix
        Z = hcluster.linkage(points, 'ward')

        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        hcluster.dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  #distribution impression in truncated branches
            )
        plt.show()
        
        
        if CLUSTER == 2:
            #Clustering with K-means algorithm:
            km = cluster.KMeans(n_clusters = n_clusters_).fit(points)
            cluster_centers = km.cluster_centers_
            labels = km.labels_
            
            clusterlon = cluster_centers[:,0]
            clusterlat = cluster_centers[:,1]
            PortsPlot(plotlon,plotlat,clusterlon,clusterlat,'K-means')
        
            # Plot result           
            fig, ax = plt.subplots(figsize=(18,6))

            colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
            for i, col in zip(range(n_clusters_), colors):
                my_members = labels == i
                cluster_center = cluster_centers[i]
                plt.plot(points[my_members, 0], points[my_members, 1], col + '.')
                plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=8)
            plt.title('Estimated number of clusters: %d (K-means)' % n_clusters_)
            plt.show()
            #Route labels are difined from port index 1, increase index
            labels = labels+1 
        else:
            #If not K-means, do Hierarchical
            labels = labelsHiarch
            x=(labels-1)
            #Find the point in each cluster that is closest to its centroid        
            centermost_points = LC.get_centermost_point(plotlon,plotlat,n_clusters_,x)
            clusterlon, clusterlat = zip(*centermost_points)    
                
            PortsPlot(plotlon,plotlat,clusterlon,clusterlat,'Hierarchical')
        
    elif CLUSTER == 3:
        #Affinity propagation clustering: STATE OF THE ART
        preference=-3
        af = cluster.AffinityPropagation(preference=preference).fit(points)
        cluster_centers = af.cluster_centers_
        labels = af.labels_
        labels_unique = np.unique(labels)
        
        #Get the number of clusters
        n_clusters_ = len(labels_unique)
        
        print('Estimated number of clusters: ' + str(n_clusters_) + \
              ' ports, from ' + str(N) + ' messages.')
        
        clusterlon = cluster_centers[:,0]
        clusterlat = cluster_centers[:,1]
        PortsPlot(plotlon,plotlat,clusterlon,clusterlat,'Affinity propagation')
        
        # Plot result        
        fig, ax = plt.subplots(figsize=(18,6))
        
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for i, col in zip(range(n_clusters_), colors):
            class_members = labels == i
            cluster_center = cluster_centers[i]
            plt.plot(points[class_members, 0], points[class_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=10)
            for x in points[class_members]:
                plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
                
        plt.title('Estimated number of clusters: %d (Affinity propagation)' % n_clusters_)
        plt.show()
        #Route labels are difined from port index 1, increase index
        labels = labels+1
        
    elif CLUSTER == 4:
        #Mean-shift clustering algorithm
        # The following bandwidth can be automatically detected using
        #bandwidth = cluster.estimate_bandwidth(points, quantile=0.2, n_samples=500)
        bandwidth = 1
        
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(points)
        cluster_centers = ms.cluster_centers_
        labels = ms.labels_
        labels_unique = np.unique(labels)
        
        #Get the number of clusters
        n_clusters_ = len(labels_unique)

        print('Estimated number of clusters: ' + str(n_clusters_) + \
              ' ports, from ' + str(N) + ' messages.')

        clusterlon = cluster_centers[:,0]
        clusterlat = cluster_centers[:,1]
        PortsPlot(plotlon,plotlat,clusterlon,clusterlat,'Mean-shift')

        # Plot result    
        fig, ax = plt.subplots(figsize=(18,6))
        
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for i, col in zip(range(n_clusters_), colors):
            my_members = labels == i
            cluster_center = cluster_centers[i]
            plt.plot(points[my_members, 0], points[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=10)
        plt.title('Estimated number of clusters: %d (Mean-shift)' % n_clusters_)
        plt.show()
        #Route labels are difined from port index 1, increase index
        labels = labels+1
    
    elif CLUSTER == 5:
        #DBSCAN
        #Number of kilometers in one radian
        kms_per_radian = 6371.0088
        #define epsilon as 10 kilometers, converted to radians for use by haversine
        eps = 20/kms_per_radian
        #Minimum number of points per cluster
        min_samples = 3
        
        dbs = cluster.DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',\
                            metric='haversine').fit(np.radians(points))
        labels = dbs.labels_
        labels_unique = np.unique(labels)
        
        #Get the number of clusters
        n_clusters_ = len(labels_unique)
        
        print('Estimated number of clusters: ' + str(n_clusters_) + \
              ' ports, from ' + str(N) + ' messages.')        

        #Find the point in each cluster that is closest to its centroid        
        centermost_points = LC.get_centermost_point(plotlon,plotlat,n_clusters_,labels)
        clusterlon, clusterlat = zip(*centermost_points)
        
        PortsPlot(plotlon,plotlat,clusterlon,clusterlat,'DBSCAN')
        
        # plot the final reduced set of coordinate points vs the original full set
        fig, ax = plt.subplots(figsize=[18, 6])
        rs_scatter = ax.scatter(clusterlon,clusterlat, c='#99cc99', edgecolor='None', alpha=0.7, s=120)
        df_scatter = ax.scatter(plotlon,plotlat, c='k', alpha=0.9, s=3)
        ax.set_title('Full data set vs DBSCAN reduced set')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
        plt.show()
        
        
        
        
        
        '''
        db = cluster.DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',\
                             metric='haversine').fit(np.radians(points))
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        
        # Plot result    
        fig, ax = plt.subplots(figsize=(18,6))
        
        # Black removed and is used for noise instead.
        labels_unique = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(labels_unique))]
        for k, col in zip(labels_unique, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = points[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)
        
            xy = points[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=2)
        
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        '''
        
        
        #Route labels are difined from port index 1, increase index
        labels = labels+1

    return clusterlat,clusterlon,labels
    
    
def Checkports(file):
    #Importing port data from LINERBIB:
    portdata = pd.DataFrame(pd.read_csv(file,sep="\t"))
    portdata['In cluster?'] = np.zeros(len(portdata))
    #Filtering out NANs from lon and lat:
    portdata = portdata[np.isfinite(portdata['Latitude']) & \
                        np.isfinite(portdata['Longitude'])]
    portdata.drop(['Draft','D_Region','Draft','CostPerFULL','CostPerFULLTrnsf', \
                   'PortCallCostFixed','PortCallCostPerFFE'], axis=1, inplace=True)    
    
    tresh = 1
    for i in range(0,len(clusterlat)):   
        for index, row in portdata.iterrows():
            if clusterlat[i] < (row['Latitude']+tresh) and \
               clusterlat[i] > (row['Latitude']-tresh) \
                and clusterlon[i] < (row['Longitude']+tresh) and \
                    clusterlon[i] > (row['Longitude']-tresh):
                    portdata['In cluster?'][index] += 1  
    
    portdata = portdata[portdata['In cluster?']>0]
    
    #Check integrity of portdata vs. cluster:                 
    if portdata['In cluster?'].sum() != n_clusters_:
        print('ERROR: Found ' + str(portdata['In cluster?'].sum()) + \
              ' ports to ' + str(n_clusters_) + ' clusters')
    
    portdata = portdata.nlargest(len(portdata),'In cluster?')
    
    return portdata
    
def ShippingNetwork(labels):
    global route
    MMSI = pd.Series(mmsi).unique()
    route = list()
    
    #Visiting sequence for each ship
    for ship in MMSI:
        r = list()
        for i in range(0,len(mmsi)):
            if mmsi[i] == ship:
                r.append(labels[i])
        r = [x[0] for x in groupby(r)]
        route.append(r)
        
    #Make a network graph:
    G=nx.Graph()
    
    for i in range(1,len(clusterlat)+1):
        G.add_node(i,pos=(clusterlon[i-1],clusterlat[i-1]))
    
    routeMatrix = np.zeros((n_clusters_+1,n_clusters_+1))
    
    for r in route:
        if len(r)>1:
            for i in range(1,len(r)):
                x = r[i-1]
                y = r[i]
                routeMatrix[x][y] = routeMatrix[x][y] + 1

    for i in range(1,len(routeMatrix[0])):
        for j in range(1,len(routeMatrix[0])):
            if ((routeMatrix[i][j] > 0) or (routeMatrix[j][i] > 0)) and (i != j):
                w = routeMatrix[i][j] + routeMatrix[j][i]
                G.add_edge(i,j,weight=w)
                routeMatrix[i][j] = 0
                routeMatrix[j][i] = 0
                if w == 0:
                    G.remove_edge(i,j)
    
    #Remove nodes with no edges
    G.remove_nodes_from(nx.isolates(G))
    
    #Visualizing the network:
    fig, ax = plt.subplots(figsize=(18,18))
    
    #Basemap worldmap
    minlon = max(-180,min(plotlon)-5) #-10
    minlat = max(-90,min(plotlat)-5) #-10
    maxlon = min(180,max(plotlon)+5) #+10
    maxlat = min(90,max(plotlat)+5) #+10
    lat0 = (maxlat+minlat)/2
    lon0 = (maxlon+minlon)/2
    lat1 = (maxlat+minlat)/2-20
        
    m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,\
                rsphere=(6378137.00,6356752.3142),\
                resolution='l',projection='cyl',\
                lat_0=lat0,lon_0=lon0,lat_ts = lat1)    
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgray',lake_color='white',zorder=0)
    
    #Draw network on map
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G,pos,alpha=0.5,node_color='r',node_size=50)
    nx.draw_networkx_edges(G,pos,alpha=0.1)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,alpha=0.2,font_size=7,clip_on=False)
    
    return route

def SpeedHistogram(speed,area):
    plt.figure()
    hist, bins = np.histogram(speeds, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

    plt.figure()
    plt.title(area)
    histval1, binsval = np.histogram(speeds, bins=50)
    histval = histval1/sum(histval1)
    width = 0.7 * (binsval[1] - binsval[0])
    center = (binsval[:-1] + binsval[1:]) / 2
    plt.bar(center, histval, align='center', width=width) #color='k'
    plt.xlabel('Speed [knots]')
    plt.ylabel('Fraction of time')
    plt.show()                
            
def PolygonAnalysis(lowtime,hightime):
    # Creating DataFram for analytical easiness           
    df = pd.DataFrame({'Speed': speeds, 'MMSI': mmsi, 'Unixtime': timestep, \
                             'Lat': plotlat, 'Lon': plotlon})

    polygons = LC.generate_polygons()
    
    #Creating empty column in the dataframe
    df['Zone'] = pd.Series({'Zone': range(len(df['MMSI']))})

    #Taking time:
    A = time.time()
    for i in range(0,len(plotlat)):
        if LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[0]):
            df.set_value(i,'Zone','Atlantic')
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[1]):
            df.set_value(i,'Zone','Pacific') #East Pacific
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[2]):
            df.set_value(i,'Zone','Pacific') #West Pacific
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[3]):
            df.set_value(i,'Zone','Indian Ocean')
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[4]):
            df.set_value(i,'Zone','Mediterranean')
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[5]):
            df.set_value(i,'Zone','North Sea')    
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[6]):
            df.set_value(i,'Zone','SE Asia')
        elif LC.point_inside_polygon(plotlon[i],plotlat[i],polygons[7]):
            df.set_value(i,'Zone','Oceania')
        else: 
            df.set_value(i,'Zone','Outside Zones')
    print('Polygon iteration time is: ' + str(time.time()-A) + ' s')      
    
    #Defining dataframes for each zone
    Atlantic = df[df['Zone'] == 'Atlantic']
    Pacific = df[df['Zone'] == 'Pacific']
    IndianOcean = df[df['Zone'] == 'Indian Ocean']
    Mediterranean = df[df['Zone'] == 'Mediterranean']
    NorthSea = df[df['Zone'] == 'North Sea']
    SEAsia = df[df['Zone'] == 'SE Asia']
    Oceania = df[df['Zone'] == 'Oceania']    
    
    Timestamps = LC.get_timevector(lowtime,hightime)
    #print(Timestamps)
    dates = [datetime.datetime.fromtimestamp(u).strftime('%d/%m/%Y') for u in Timestamps]
    #print(dates)

    
    #Filter data for monthly basis
    A = time.time()
    speeds_monthly_Atlantic,unique_vessels_monthly_Atlantic,interval_Atlantic \
        = LC.monthly_filter(Atlantic,Timestamps)
    speeds_monthly_Pacific,unique_vessels_monthly_Pacific,interval_Pacific \
        = LC.monthly_filter(Pacific,Timestamps)
    speeds_monthly_IndianOcean,unique_vessels_monthly_IndianOcean,interval_IndianOcean \
        = LC.monthly_filter(IndianOcean,Timestamps)
    speeds_monthly_Medittarnean,unique_vessels_monthly_Medittarnean,interval_Medittarnean \
        = LC.monthly_filter(Mediterranean,Timestamps)
    speeds_monthly_NorthSea,unique_vessels_monthly_NorthSea,interval_NorthSea \
        = LC.monthly_filter(NorthSea,Timestamps)
    speeds_monthly_SEAsia,unique_vessels_monthly_SEAsia,interval_SEAsia \
        = LC.monthly_filter(SEAsia,Timestamps)
    speeds_monthly_Oceania,unique_vessels_monthly_Oceania,interval_Oceania \
        = LC.monthly_filter(Oceania,Timestamps)
    speeds_monthly_World,unique_vessels_monthly_World,interval_World \
        = LC.monthly_filter(df,Timestamps)
    print('Monthly filtering time is: ' + str(time.time()-A) + ' s') 
    
    #Percentage of fleet observed in zone each month
    prcentage_Atlantic = LC.percentageMonthly(unique_vessels_monthly_Atlantic,
                                              unique_vessels_monthly_World)
    prcentage_Pacific = LC.percentageMonthly(unique_vessels_monthly_Pacific,
                                              unique_vessels_monthly_World)
    prcentage_IndianOcean = LC.percentageMonthly(unique_vessels_monthly_IndianOcean,
                                              unique_vessels_monthly_World)
    prcentage_Medittarnean = LC.percentageMonthly(unique_vessels_monthly_Medittarnean,
                                              unique_vessels_monthly_World)
    prcentage_NorthSea = LC.percentageMonthly(unique_vessels_monthly_NorthSea,
                                              unique_vessels_monthly_World)
    prcentage_SEAsia = LC.percentageMonthly(unique_vessels_monthly_SEAsia,
                                              unique_vessels_monthly_World)
    prcentage_Oceania = LC.percentageMonthly(unique_vessels_monthly_Oceania,
                                              unique_vessels_monthly_World)
        
    #Plotting the vessel distribution    
    timeperiod = list(range(0,(len(Timestamps)-1)))
    plt.style.use('bmh')
    
    #Plotting number of unique vessels
    fig, ax = plt.subplots(figsize=(15,6))
    plt.title('Number of unique vessels in zone')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Vessels in zone')
    ax.plot(timeperiod,unique_vessels_monthly_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,unique_vessels_monthly_Pacific,label = 'Pacific')
    ax.plot(timeperiod,unique_vessels_monthly_IndianOcean, label = 'Indian Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_Medittarnean, label = 'Mediettarnean Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_NorthSea,label = 'North Sea')
    ax.plot(timeperiod,unique_vessels_monthly_SEAsia, label = 'South East Asia')
    ax.plot(timeperiod,unique_vessels_monthly_Oceania, label = 'Oceania')
    ax.plot(timeperiod,unique_vessels_monthly_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()
    
    #Percentage of fleet observed in zone each month
    fig, ax = plt.subplots(figsize=(15,6))
    plt.title('Percentage of fleet observed in zone each month')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Percentage [%]')
    ax.plot(timeperiod,prcentage_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,prcentage_Pacific,label = 'Pacific')
    ax.plot(timeperiod,prcentage_IndianOcean, label = 'Indian Ocean')
    ax.plot(timeperiod,prcentage_Medittarnean, label = 'Mediettarnean Ocean')
    ax.plot(timeperiod,prcentage_NorthSea,label = 'North Sea')
    ax.plot(timeperiod,prcentage_SEAsia, label = 'South East Asia')
    ax.plot(timeperiod,prcentage_Oceania, label = 'Oceania')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()    
    
    
    #Plotting mean speed of vessels
    fig, ax = plt.subplots(figsize=(15,6))
    plt.title('Average speed per month per zone')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Speed [knots]')
    ax.plot(timeperiod,speeds_monthly_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,speeds_monthly_Pacific,label = 'Pacific')
    ax.plot(timeperiod,speeds_monthly_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show() 
    
    #Plotting mean message interval of vessels
    fig, ax = plt.subplots(figsize=(15,6))
    plt.title('Average message interval')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Time [hours]')
    #ax.plot(timeperiod,interval_Atlantic,label = 'Atlantic')
    #ax.plot(timeperiod,interval_Pacific,label = 'Pacific')
    ax.plot(timeperiod,interval_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()  
    
    #SPeed histogram
    SpeedHistogram(Atlantic['Speed'].tolist(),'Atlantic')
    SpeedHistogram(Pacific['Speed'].tolist(),'Pacific')
    
    return df
   
    # Statistical Data
#   AtlanticMean = st.mean(AtlanticSpeed)
#   AtlanticVar = st.stdev(AtlanticSpeed)
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

def DraughtHistogram():    
    plt.figure()
    histval1, binsval = np.histogram(draught, bins=20)
    histval = histval1/sum(histval1)
    width = 0.7 * (binsval[1] - binsval[0])
    center = (binsval[:-1] + binsval[1:]) / 2
    plt.bar(center, histval, align='center', width=width)
    plt.xlabel('Draught [meters]')
    plt.ylabel('Fraction of time')
    plt.show()
    
################################################
## THE FOLLOWING PARTS PLOTS POSITIONS ON MAPS #
################################################
def LocalMap():
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

    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgray',lake_color='white',zorder=0) #,zorder=0
    x, y = m(plotlon,plotlat) 
    #Ships:
    data = m.scatter(x,y,0.01,marker='o',c='black',label = 'AIS Data w/ speed > 5 knots and navigational status 1 or 5')
    #m.drawcoastlines()
    #plt.legend(handles=[data])
    #plt.title('Ship movements')
    #m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])       
    #m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
    #m.bluemarble()
    #plt.saveffig('/Users/PatrickAndreNaess/Desktop/PyPlots/current_plot.eps',\
    #            format='eps', dpi=1000)
   
################################################
## THE FOLLOWING PART PLOTS SPEED AGAINST TIME #
################################################

def SpeedDev(): 
    Timesteps = list()
    for i in range(0,len(timestep)):
        unixconv = datetime.datetime.fromtimestamp(timestep[i])
        Timesteps.append(unixconv)
 
    plt.figure()
    plt.plot(Timesteps,speeds,'k')
    plt.show()
    plt.ylabel("Speed [knots]")   
    
def DraughtInterpolate():    
    global IntDraught    
    IntDraught = list()
    draughtSpeed = list()
    draughtIntTime = list()
    for i in range(0,len(timestep)):
        
        if timestep[i] >= draughttime[0]:
            idx = (np.abs(np.array(draughttime)-timestep[i])).argmin()
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
        Unixconv = datetime.datetime.fromtimestamp(AnTime[i])
        AnSteps.append(Unixconv)

################################################
## THE FOLLOWING PART PLOTS INTERVAL FREQUENCY #
################################################  
        
    Count = list()
    Number = list()
    for i in range(0,360):
        Count.append(i)
        Number.append(AnDiff.count(i))
    
    plt.figure(figsize=(18,6))
    plt.plot(Count,Number)
    plt.show()   
    
    plt.figure(figsize=(18,6))
    plt.scatter(AnSteps,AnSpeed,c='k',s=0.1)
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
        Unixconv = datetime.datetime.fromtimestamp(AtTime[i])
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

