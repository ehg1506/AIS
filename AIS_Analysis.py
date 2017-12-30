#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:45:53 2017

@author: PatrickAndreNaess
"""

import sqlite3
import matplotlib.pyplot as plt
import time
import datetime
from mpl_toolkits.basemap import Basemap
import numpy as np
from scipy import stats
import pandas as pd
import sklearn.cluster as cluster
import scipy.cluster.hierarchy as hcluster
import loc_check as LC
import networkx as nx
from itertools import cycle, groupby
from pylab import boxplot
import matplotlib.colors as mcolors


#######################################################
## THE FOLLOWING PART EXTRACT DATA FROM THE DATABASE ##
#######################################################

def ExtractData(filepath,a,b,c,d,lowtime,hightime,maxspeed,minspeed,Analysis):
    global draughttime
    global useridDraught
    global draught
    global plotlat
    global plotlon
    global speeds
    global timestep
    global mmsi
    global navstat
    
    speeds = list()
    plotlat = list()
    plotlon = list()
    timestep = list()
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
        str(maxspeed),str(lowtime),str(hightime),'')
    
    # Extract data from database:
    A = time.time()
    with conn:
        cur = conn.cursor()  
        if Analysis == 1:
            cur.execute(SQLstring1)
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
        unixtime >= %s and unixtime <= %s" % ('Panamax5',str(lowtime),\
        str(hightime))
    
    # Extract draught data from database:
    with conn:
        cur = conn.cursor()
        if Analysis == 1:
            cur.execute(SQLstring2)
        draughtdata = cur.fetchall()
    
    for i in range(0,len(draughtdata)):
        draughtstrip = draughtdata[i]
        if draughtstrip[1]/10 > 5:
            draughttime.append(draughtstrip[0])
            draught.append(draughtstrip[1]/10)
            useridDraught.append(draughtstrip[2])
            
    print('Database extraction time is: ' + str(time.time()-A) + ' s')
    
    return plotlon, plotlat

##########################################################
## THE FOLLOWING PART PLOTS CLUSTERS AGAINTS DATAPOINTS ##
##########################################################

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

###############################################
## THE FOLLOWING PART CLUSTER PORT LOCATIONS ##
###############################################

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
    
    CLUSTER = 4
    
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
        
        #Route labels are difined from port index 1, increase index
        labels = labels+1
        #Nois get index 0

    return clusterlat,clusterlon,labels

#########################################################
## THE FOLLOWING PART GEO-FENCE EACH PORT TO FIND FLOW ##
######################################################### 

def GeoFencePorts():    
    global clusterlat
    global clusterlon
    global n_clusters_
    labels = list()
    
    #Found Clusters:
    clusterlat = np.array([-39.47005546, -43.60831855, -34.77054182,  21.31065826,
       -37.66173421, -33.97062466, -37.81408942,  -2.30858274,
         9.29704122,  33.74185717,  37.79363253,  47.4857799 ,
        24.99791398, -27.31810546,  53.90198618,  44.66745021,
       -33.90899305, -12.04922705, -29.76686978,   6.40452044,
       -33.37102237,   6.22932479,  40.64020991, -32.21156226,
        35.44804364, -45.81215551, -36.84238701,  14.68634337,
       -26.42434145,  18.8835651 , -24.1039243 ,  18.42304778,
        21.45734532, -32.04233804,   4.73937117, -20.14112014,
        49.03351612,  -4.76633078,  28.13594186,  -3.49574528,
        19.06536673,  10.39705054,  10.59979919, -46.59281815,
       -44.38887017,  36.88249897,  29.94976019, -20.20391488,
        -5.08124783,  31.20420159,   1.33428887, -22.97416078,
        13.89098019, -34.62130546,  -8.39160417,   6.97603545,
        22.73516487,  19.2078054 ,  17.87199473, -36.88916047,
        45.58785705,  -8.75631341,  22.47307654,  22.2183688 ,
        24.78724314,  38.975195  ,  40.27725702,  42.77130339,
         2.9424092 ,  32.12612962,  34.68089737, -33.8377804 ,
       -12.9681302 ,  42.96487848, -34.94671111,  10.47932094,
       -41.27914292,   3.89375714,  11.59775551,  13.44109474,
        39.48133462,  35.02566472,   7.29433333,  26.52761833,
        46.32992778, -22.80078606, -20.93161067, -41.14005833,
        -6.07800383,  51.45372717,  38.94386667,  16.94442259,
         5.61572926,   5.27301352,  42.34259981,  44.73004037,
        35.01565208,  20.99669208,   9.97678583,  31.853125  ,
        32.71033286,  51.343125  ,  14.56655722,  54.28542667,
       -18.47346292,  24.36345   ,   5.42295208,  27.07394167,
        26.50487083,  37.93383083,   0.07519083, -19.48872167,
        13.09184667,  15.19428111,  31.83400222,  29.61256667,
        39.25273556,  22.53841389,  39.21680111,  -1.53474889,
        25.76633333,  26.27615333,  12.80466667,  28.57381667,
        29.481     ,  13.03665417,  39.97689333,  42.99483667,
        49.47274333,  29.911615  ,  16.47556667,  39.89974333,
        22.65683333,  27.54949333,  17.88768   ,  34.74833333,
       -19.961925  ,  50.909475  ,  36.141     ,  17.69337333,
        19.60073   ,  51.4475    , -23.06601833,  36.0015    ,
        28.17988667,  21.67966667,  11.91631667,  40.96442333,  13.44648167])
    clusterlon = np.array([  1.76922655e+02,   1.72726930e+02,   1.38482545e+02,
        -1.57876474e+02,   1.76175671e+02,   1.51213697e+02,
         1.44912802e+02,  -7.99490750e+01,  -7.98278532e+01,
        -1.18246320e+02,  -1.22308069e+02,  -1.22366113e+02,
         5.50608983e+01,   1.53176346e+02,  -1.66529019e+02,
        -6.36264924e+01,   1.84484570e+01,  -7.71563516e+01,
         3.11199504e+01,   3.06170263e+00,  -7.16254414e+01,
         1.79391882e+00,  -7.40775162e+01,  -5.19802947e+01,
         1.39744361e+02,   1.70626450e+02,   1.74783949e+02,
        -1.74269819e+01,  -4.85186955e+01,   7.28024397e+01,
        -4.63046993e+01,  -6.96300694e+01,   3.91459508e+01,
         1.15742993e+02,  -6.61855168e+00,   5.74915478e+01,
        -1.23147928e+02,   1.17964966e+01,  -1.54135653e+01,
        -3.87771226e+01,  -1.04316885e+02,  -7.55326580e+01,
        -6.16260588e+01,   1.68335412e+02,   1.71256825e+02,
        -7.61133508e+01,   3.25255831e+01,  -7.01559235e+01,
        -8.11085425e+01,   3.22465814e+01,   1.03859839e+02,
        -4.31378616e+01,  -9.07828569e+01,  -5.83031861e+01,
        -3.49479663e+01,   7.98073635e+01,   6.97138650e+01,
        -9.61273605e+01,  -7.68576118e+01,  -7.31283614e+01,
        -7.35055777e+01,   1.32739070e+01,  -9.78842187e+01,
         9.17244111e+01,   6.71337195e+01,   1.17896560e+02,
         1.22048324e+02,   1.32933254e+02,   1.01311060e+02,
        -8.11370765e+01,   1.35284958e+02,   2.56906563e+01,
        -3.85220660e+01,   1.44323937e+02,  -5.61464100e+01,
        -6.80043917e+01,   1.74790112e+02,  -7.70530183e+01,
         4.31076456e+01,   7.24813317e+01,  -2.95758462e-01,
         1.36808730e+02,   1.25716667e+02,  -7.87658733e+01,
         3.06626625e+01,   1.44654917e+01,   5.53179023e+01,
         1.46852689e+02,   1.06901232e+02,   4.89637500e-01,
         1.21959517e+02,   5.40187135e+01,   1.14992593e-02,
        -4.01374259e+00,  -7.10202191e+01,   3.77874319e+01,
         1.28939719e+02,   7.20745258e+01,   7.62506677e+01,
        -1.16629742e+02,  -7.97462990e+01,   4.27161905e+00,
         1.20912372e+02,  -1.30361132e+02,  -7.03380646e+01,
         1.24080112e+02,   1.00360642e+02,   5.61306838e+01,
         5.01967708e+01,  -8.84884667e+00,  -5.09953483e+01,
         4.41750900e+01,   8.03193222e+01,   1.45677639e+02,
         3.46493611e+01,  -9.50077389e+01,  -7.65446611e+01,
         1.20300958e+02,   9.06839556e+00,  -4.87512189e+01,
        -8.01560000e+01,   1.27901382e+02,   4.49971400e+01,
        -4.69219000e+01,  -4.32889167e+01,   1.00871785e+02,
         1.19827951e+02,   1.31651461e+02,   1.53000000e-01,
        -9.01084617e+01,   1.20277472e+02,  -7.51352433e+01,
         1.13678333e+02,   5.25399850e+01,  -1.02207068e+02,
         1.19436000e+02,   3.55639983e+01,  -1.46385833e+00,
        -5.43583333e+00,   8.32997583e+01,   3.72362317e+01,
         2.67283333e+00,  -7.03869283e+01,   1.20214167e+02,
         4.86997133e+01,   1.08638167e+02,   1.01930700e+02,
         2.86824200e+01,   1.44630170e+02])
    
    n_clusters_ = len(clusterlat)
    
    df = pd.DataFrame({'Lat': plotlat, 'Lon': plotlon})
    df['Label'] = np.zeros(len(plotlat))
    
    #Taking time:
    A = time.time()    
    
    tresh = 0.5
    for j in range(0,len(plotlat)):
        x = 0
        for i in range(0,len(clusterlat)): 
            if (plotlat[j] < clusterlat[i]+tresh) and (plotlat[j] > clusterlat[i]-tresh) \
            and (plotlon[j] < clusterlon[i]+tresh) and (plotlon[j] > clusterlon[i]-tresh):
                if x > 0:
                    x = 1
                else:    
                    labels.append(i)
                    x = 1
        if x == 0:
            labels.append(-1)

    print('Port geo-fence time is: ' + str(time.time()-A) + ' s')
    #Route labels are difined from port index 1, increase index
    l = np.array(labels)
    labels = l+1                    
                        
    return labels

##############################################################
## THE FOLLOWING PART CHECK PORT LOCATIONS AGAINST LINERLIB ##
##############################################################    
    
def Checkports(file):
    #Importing port data from LINERBIB:
    portdata = pd.DataFrame(pd.read_csv(file,sep="\t"))
    portdata['In cluster?'] = np.zeros(len(portdata))
    #Filtering out NANs from lon and lat:
    portdata = portdata[np.isfinite(portdata['Latitude']) & \
                        np.isfinite(portdata['Longitude'])]
    portdata.drop(['Cabotage_Region','Draft','D_Region','Draft','CostPerFULL',\
                   'CostPerFULLTrnsf', 'PortCallCostFixed','PortCallCostPerFFE']
                   ,axis=1, inplace=True)    
    
    tresh = 0.5
    for i in range(0,len(clusterlat)):   
        for index, row in portdata.iterrows():
            if clusterlat[i] < (row['Latitude']+tresh) and \
               clusterlat[i] > (row['Latitude']-tresh) \
                and clusterlon[i] < (row['Longitude']+tresh) and \
                    clusterlon[i] > (row['Longitude']-tresh):
                    
                        portdata['In cluster?'][index] += 1  
    
    #Check integrity of portdata vs. cluster:                 
    if portdata['In cluster?'].sum() != n_clusters_:
        print('ERROR: Found ' + str(portdata['In cluster?'].sum()) + \
              ' ports to ' + str(n_clusters_) + ' clusters')
    
    portdata = portdata.nlargest(len(portdata),'In cluster?')
    
    return portdata
 
##################################################
## THE FOLLOWING PART GENERATE SHIPPING NETWORK ##
##################################################    

def ShippingNetwork(labels):
    global route
    MMSI = pd.Series(mmsi).unique()
    route = list()
    
    #Visiting sequence for each ship
    for ship in MMSI:
        r = list()
        for i in range(0,len(mmsi)):
            if mmsi[i] == ship and labels[i]>0: #0 is noise, ignore noise
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
                if w < 3:
                    G.remove_edge(i,j)
    
    print('Found '+str(G.size(weight='weight'))+' port-to-port routes from ' + str(len(plotlat)) + ' messages.')
    
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
    
    ############################################
    ## THE FOLLOWING PART ANALYSE THE NETWORK ##
    ############################################
    
    #Draw for network analysis
    def draw(G, pos, measures, measure_name):
        fig, ax = plt.subplots(figsize=(16,5))
        nodes = nx.draw_networkx_nodes(G, pos, node_size=70, cmap=plt.cm.plasma, 
                                       node_color=list(measures.values()),
                                       nodelist=list(measures.keys()),alpha=0.85)
        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
        nx.draw_networkx_edges(G, pos,alpha=0.1)
        plt.title(measure_name)
        plt.colorbar(nodes)
        plt.axis('off')
        plt.show()
    
    # Degree Centrality:
    draw(G, pos, nx.degree_centrality(G), 'Degree Centrality')
    
    # with Degree Distribution
    degree_sequence=sorted(nx.degree(G).values(),reverse=True)   
    plt.loglog(degree_sequence,'b-',marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
    
    # Eigenvector Centrality:
    draw(G, pos, nx.eigenvector_centrality(G), 'Eigenvector Centrality')
    
    # Closeness Centrality:
    draw(G, pos, nx.closeness_centrality(G), 'Closeness Centrality')
    
    # Betweenness Centrality:
    draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')
    
    return route,G

##################################################
## THE FOLLOWING PART PLOTS HISTOGRAM FOR SPEED ##
##################################################

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
    plt.bar(center, histval, align='center', width=width)
    plt.xlabel('Speed [knots]')
    plt.ylabel('Fraction of time')
    plt.show()                
    
##################################################
## THE FOLLOWING PART DOES THE POLYGON ANALYSIS ##
##################################################
        
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
    speeds_monthly_Atlantic,monthly_stdev_Atlantic,unique_vessels_monthly_Atlantic,interval_Atlantic \
        = LC.monthly_filter(Atlantic,Timestamps)
    speeds_monthly_Pacific,monthly_stdev_Pacific,unique_vessels_monthly_Pacific,interval_Pacific \
        = LC.monthly_filter(Pacific,Timestamps)
    speeds_monthly_IndianOcean,monthly_stdev_IndianOcean,unique_vessels_monthly_IndianOcean,interval_IndianOcean \
        = LC.monthly_filter(IndianOcean,Timestamps)
    speeds_monthly_Medittarnean,monthly_stdev_Mediterranean,unique_vessels_monthly_Medittarnean,interval_Medittarnean \
        = LC.monthly_filter(Mediterranean,Timestamps)
    speeds_monthly_NorthSea,monthly_stdev_NorthSea,unique_vessels_monthly_NorthSea,interval_NorthSea \
        = LC.monthly_filter(NorthSea,Timestamps)
    speeds_monthly_SEAsia,monthly_stdev_SEAsia,unique_vessels_monthly_SEAsia,interval_SEAsia \
        = LC.monthly_filter(SEAsia,Timestamps)
    speeds_monthly_Oceania,monthly_stdev_Oceania,unique_vessels_monthly_Oceania,interval_Oceania \
        = LC.monthly_filter(Oceania,Timestamps)
    speeds_monthly_World,monthly_stdev_World,unique_vessels_monthly_World,interval_World \
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
    fig, ax = plt.subplots(figsize=(12,5))
    #plt.title('Number of unique vessels in zone')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Vessels in zone')
    ax.plot(timeperiod,unique_vessels_monthly_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,unique_vessels_monthly_Pacific,label = 'Pacific')
    ax.plot(timeperiod,unique_vessels_monthly_IndianOcean, label = 'Indian Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_Medittarnean, label = 'Mediterranean Ocean')
    ax.plot(timeperiod,unique_vessels_monthly_NorthSea,label = 'North Sea')
    ax.plot(timeperiod,unique_vessels_monthly_SEAsia, label = 'South East Asia')
    ax.plot(timeperiod,unique_vessels_monthly_Oceania, label = 'Oceania')
    ax.plot(timeperiod,unique_vessels_monthly_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=4,loc=3,mode="expand",borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()
    
    #Percentage of fleet observed in zone each month
    fig, ax = plt.subplots(figsize=(12,5))
    #plt.title('Percentage of fleet observed in zone each month')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Percentage [%]')
    ax.plot(timeperiod,prcentage_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,prcentage_Pacific,label = 'Pacific')
    ax.plot(timeperiod,prcentage_IndianOcean, label = 'Indian Ocean')
    ax.plot(timeperiod,prcentage_Medittarnean, label = 'Mediterranean Ocean')
    ax.plot(timeperiod,prcentage_NorthSea,label = 'North Sea')
    ax.plot(timeperiod,prcentage_SEAsia, label = 'South East Asia')
    ax.plot(timeperiod,prcentage_Oceania, label = 'Oceania')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=4,loc=3,mode="expand",borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()    
    
    #Plotting mean speed of vessels
    fig, ax = plt.subplots(figsize=(12,5))
    #plt.title('Average speed per month per zone')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Speed [knots]')
    ax.plot(timeperiod,speeds_monthly_Atlantic,label = 'Atlantic')
    ax.plot(timeperiod,speeds_monthly_Pacific,label = 'Pacific')
    ax.plot(timeperiod,speeds_monthly_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=3,loc=3,mode="expand",borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show() 
    
    #Plotting world speed with standard deviation
    fig, ax = plt.subplots(figsize=(12,5))
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Speed [knots]')
    boxplot(monthly_stdev_World,0,'')
    plt.show() 
    
    #Plotting mean message interval of vessels
    fig, ax = plt.subplots(figsize=(12,5))
    #plt.title('Average message interval')
    plt.xlabel ('Months from %s to %s' % (str(dates[0]),str(dates[len(dates)-1])))
    plt.ylabel('Time [hours]')
    #ax.plot(timeperiod,interval_Atlantic,label = 'Atlantic')
    #ax.plot(timeperiod,interval_Pacific,label = 'Pacific')
    ax.plot(timeperiod,interval_World, label = 'World')
    #Placeing the legend outside the plot box:
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2,loc=3,mode="expand",borderaxespad=0.)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_lines():
        label.set_linewidth(1)
    plt.show()  
    
    #SPeed histogram
    SpeedHistogram(Atlantic['Speed'].tolist(),'Atlantic')
    SpeedHistogram(Pacific['Speed'].tolist(),'Pacific')

    LC.plot_inside_polygons(Atlantic['Lon'].tolist(),Atlantic['Lat'].tolist())  
    
    return df        
    
######################################################
## THE FOLLOWING PART PLOTS HISTOGRAM OF SPEED DIST ##
######################################################

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
## THE FOLLOWING PARTS PLOTS POSITIONS ON MAP ##
################################################
    
def LocalMap():
    minlon = max(-180,min(plotlon)-5) #-5
    minlat = max(-90,min(plotlat)-5) #-5
    maxlon = min(180,max(plotlon)+5) #+5
    maxlat = min(90,max(plotlat)+5) #+5
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
    m.scatter(x,y,0.01,marker='o',c='black')
    #m.drawcoastlines()
    #plt.legend(handles=[data])
    #plt.title('Ship movements')
    #m.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])       
    #m.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
    #m.bluemarble()
    #plt.saveffig('/Users/PatrickAndreNaess/Desktop/PyPlots/current_plot.eps',\
    #            format='eps', dpi=1000)                 

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
