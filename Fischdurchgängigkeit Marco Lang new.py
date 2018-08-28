#----------------------------------------------------------------------------------------------
#   IMPORTS
#----------------------------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import shapefile
import numpy as np
from numpy import random
import re
import os
import sys
from scipy.spatial import distance
import arcpy
import shutil
import copy
import pandas as pd
import xlsxwriter
import csv
import re


#----------------------------------------------------------------------------------------------
#   INPUTS
#----------------------------------------------------------------------------------------------
basepath = "D:/Studium/Master/Masterarbeit/Model Seminar/Guerbe/BASEMENT/guerbe_ausschnitt_networkx_new"
with open(basepath+'/guerbe.edg','r') as edg:
    lines=edg.readlines()  # type: List[str]
nodeshapefile = shapefile.Reader(basepath+'/guerbemesh_Quality_nodes_new_Renumbered_nodes.shp')
depthsol = open(basepath+'/guerbe_nds_depth.sol','r').read().strip('\n').split()
runoff = []
with open(basepath+'/hydrograph.txt', 'r') as hydrograph:
    reader = csv.reader(hydrograph, delimiter='\t')
    for row in reader:
        print row[1]
        runoff.append(row[1])

runoff=runoff[1:]


#----------------------------------------------------------------------------------------------
#   CREATE A PERSONAL GDB
#----------------------------------------------------------------------------------------------
## Set local variables
###gdbout_folder_path = basepath
###gdbout_name_gdb = "seminararbeit.gdb"

## Execute CreatePersonalGDB
###arcpy.CreatePersonalGDB_management(gdbout_folder_path, gdbout_name_gdb)

#   Inputs
#Gdb=basepath+'/seminararbeit.gdb'

## Use FeatureClassToGeodatabase to copy feature class nds
###in_features_nds = basepath+'/guerbemesh_Quality_nodes_new_Renumbered_nodes.shp'
###out_location_nds = Gdb

## Execute FeatureClassToGeodatabase
###arcpy.FeatureClassToGeodatabase_conversion(in_features_nds, out_location_nds)

## Use FeatureClassToGeodatabase to copy feature class els
###in_features_els = basepath+'/guerbemesh_Quality_nodes_new_Renumbered_elements.shp'
###out_location_els = Gdb

## Execute FeatureClassToGeodatabase
###arcpy.FeatureClassToGeodatabase_conversion(in_features_els, out_location_els)


#----------------------------------------------------------------------------------------------
#   CREATE A FUTURE DATASET
#----------------------------------------------------------------------------------------------
## Set local variables
###out_dataset_path = Gdb
###out_name_fd = "analysisresults"

## Creating a spatial reference object
###sr = arcpy.SpatialReference(basepath+'/guerbemesh_Quality_nodes_new_Renumbered_nodes.prj')

# Execute CreateFeaturedataset
#arcpy.CreateFeatureDataset_management(out_dataset_path, out_name_fd,sr)


#----------------------------------------------------------------------------------------------
#   CREATE A FOLDER FOR CSV FILES
#----------------------------------------------------------------------------------------------
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


## Example
###createFolder(basepath+'/csv_files')

#   Inputs
depthvaluesarray_transpose_nodes_csv=basepath+'/csv_files/depthvaluesarray_transpose_nodes.csv'
depthvaluesarray_transpose_csv=basepath + "/csv_files/depthvaluesarray_transpose.csv"


#----------------------------------------------------------------------------------------------
#   CREATE THE GRAPH
#----------------------------------------------------------------------------------------------
## Create an empty graph with no nodes and no edges
G = nx.Graph()

## The data, which we need to calculate are only from the sixth line. That's why we skip the lines in front of it
edgdata = lines[5:]

## The lines are separated by a whitespace, so you can split them into rows on this whitespace
columns=[line.split() for line in edgdata]

## Because the columns are of data type integer, they must be converted into a float
Edge = [int(column[0]) for column in columns]
Node1 = [int(column[1]) for column in columns]

## Make a copy of Node1 while expanding the list of Node2 values further down
Elem_L = [int(column[3]) for column in columns]
Elem_R = [column[4] for column in columns]
Node1copy=Node1
Node2 = [int(column[2]) for column in columns]

## In the Elem_R series, the NULL values are replaced with -9999.
Elem_R = [rep.replace('NULL', '-9999')for rep in Elem_R]

## Now the list of Node2 is extended by the copied list Node1copy.
Node1copy.extend(Node2)

## Convert strings in List Elem_R to integer
Elem_R=map(int,Elem_R)

## Add leading zeros to my data
[str(item).zfill(5) for item in Edge]

## To create edges, the two node rows must be combined in a single tuple. => Edges=(Node1_line1,Node2_line1),(Node1_line2,Node2_line2),(Node1_line3,Node2_line3)
tuple_N = zip(Node1, Node2)
print tuple_N

## Setting Node1 back to the original list
Node1 = [int(column[1]) for column in columns]
print nx.info(G)

## Creating Edges and check the graph info
G.add_edges_from(tuple_N)
print nx.info(G)

## For the location of the nodes, the shapefile with the nodes before the basement simulation is used
fields = nodeshapefile.fields[1:]
fields_name = [field[0] for field in fields] #-> the fields names in a list
###print fields_name
attributes = nodeshapefile.records()
print attributes


## Extract the nested list and get a simplier list
Simple_Attributlist=[]
for inner_l in attributes:
    for item in inner_l:
        Simple_Attributlist.append(item)

## Extract nodes, x- and y-position from attribute table from shapefile
Attribute_nodes = Simple_Attributlist[0::4]
Attribute_xPos = Simple_Attributlist [1::4]
Attribute_yPos = Simple_Attributlist[2::4]
Attribute_pos = zip(Attribute_xPos,Attribute_yPos)
Attribute_pos_ID = zip(Attribute_pos,Attribute_nodes)

## Adding nodes to Graph
G.add_nodes_from(Attribute_nodes)
print nx.info(G)

## Creat an empty dictionary
emd={}

## Fill dictionary with nodes as keys and xy-position as values
emd=dict(zip(Attribute_nodes,Attribute_pos))

nx.draw(G,pos=emd, node_size=10)
#nx.draw_networkx(G,pos=emd, node_size=10, with_labels=True)


#----------------------------------------------------------------------------------------------
#   ADDING ATTRIBUTE WATER DEPTH TO THE NODES
#----------------------------------------------------------------------------------------------
## It would also be possible to add the velocity and the water surface elevation next to the water depth
## Create outer dictionary with TS1 - TS11 as keys
Steps=range(1,12)
Timesteps=[]
for i in Steps:
    Timesteps.append("TS"+str(i))
    i=i+1
TS_tuple=tuple(Timesteps)

Depth_TS_out={}
Depth_TS_out=dict.fromkeys(Timesteps)

## Integrating above code into for loop with splitting .sol-file into timesteps and add them to the subdictionarys from 'Depth_TS_out'
outputBase = 'Depth_TS'
splitLen = 4086


depthvalues=[]
i=3
k=4088
for line in depthsol:
    if line.startswith("TS"):
        outputData = (depthsol[i:k])
        depthvalues.append(outputData)
        i=i+1
        k=k+1
    else:
        i=i+1
        k=k+1


## Converting string in depthvalues to float
depthvalues_float=[]
for parts in depthvalues:
    depthvalues_float.append([float(q) for q in parts])

depthvalues_tuple=tuple(depthvalues_float)

## Creating dictionary with timesteps as keys and depthvalues as values because networkx needs one
Depth_TS={}
Depth_TS=dict(zip(TS_tuple,depthvalues_tuple))

## Adding Depth_TS as attributes to the graph
for timesteps in Depth_TS:
    nx.set_node_attributes(G, Depth_TS[timesteps], 'Water_Depth')

nx.draw(G,pos=emd, node_size=10)
#nx.draw_networkx(G,pos=emd, node_size=10, with_labels=True)


#----------------------------------------------------------------------------------------------
#   TOOL 1: CALCULATE THE SHORTEST PATH FOR DIFFERENT OUTFLOWS
#----------------------------------------------------------------------------------------------
## Converting list with water depths to array to calculate the fish paths
## Make Loop which takes from the directory Depth_TS each list (each timestep) and calculates if there is a Path. each timestep should then represent a different outflow
## Empty list with all timesteps shortestpath True and False
## The loop that automates the function is still missing but it works manually!!!

shortestpathlistrunoff = []

# list
Depth_TS_List = []

for value in Depth_TS:
    releases= Depth_TS[value]
    Depth_TS_List.append(releases)

i=0
while i <= 10:
    ## Timestep
    lastTS = Depth_TS_List[i] #change the time step in the bracket so that each time step is executed

    ## timestep only with the dry nodes
    lastTSarraysmall = np.asarray(lastTS)
    lastTS_transposesmall = lastTSarraysmall.transpose()
    water_pointssmall = np.where(lastTS_transposesmall<=0.2)
    water_points_array_small = np.asarray(water_pointssmall)
    water_points_transpose_small = water_points_array_small.transpose()

    ## matrix
    smallvaluelist = water_points_array_small.tolist()
    ## list
    smalldepthnodes = smallvaluelist[0]
    ## Increase each number in the list by one because the nodes begin with 1 instead of 0
    smalldepthnodes = [int(n+1) for n in smalldepthnodes]
    smalldepthnodes

    ## Find all edges incident on the nodes smaller than 0.2m - node pairs
    smalldepthedges = G.edges(smalldepthnodes)

    S = nx.Graph()
    S.add_nodes_from(smalldepthnodes)
    S.add_edges_from(smalldepthedges)
    S.number_of_nodes()
    S.number_of_edges()

    ## The smalldepthnodes must removed from the Graph to get the right graph
    Sub = nx.Graph(G)
    Sub.remove_nodes_from(smalldepthnodes)
    Sub.number_of_nodes()
    Sub.number_of_edges()

    ## search 2 nodes which are in the graph
    search2nodes = Sub.nodes
    if not search2nodes:
        print("List is empty")
        element = 0
        shortestpathlistrunoff.append(element)
    else:
        source = min(search2nodes)
        target = max(search2nodes)

        ## Find shortest path
        nx.draw(Sub, emd, node_color='k')
        plt.axis('equal')
        nx.has_path(Sub, source, target)
        shortestpath = nx.has_path(Sub, source, target)

        ## draw path in blue
        # path = nx.shortest_path(Sub,source,target)
        # path_edges = zip(path,path[1:])
        # nx.draw_networkx_nodes(Sub,emd,nodelist=path,node_color='b')
        # nx.draw_networkx_edges(Sub,emd,edgelist=path_edges,edge_color='b',width=10)
        # plt.axis('equal')
        # plt.show()

        ## List with all timesteps shortestpath True and False
        shortestpathlist = [shortestpath]
        for element in shortestpathlist:
            if element is True:
                element = 1
                shortestpathlistrunoff.append(element)
            else:
                element = 0
                shortestpathlistrunoff.append(element)
    i = i + 1
print "loop 1 done"

shortestpathlistrunoff.sort()

plt.title('Kuerzester Weg fuer verschiedene Abfluesse')
plt.xlabel('Abfluss [m^3/s]')
plt.ylabel('Durchgaengikeit')
plt.ylim(-0.05,1.05)
plt.grid(True)
plt.plot(runoff,shortestpathlistrunoff,runoff,shortestpathlistrunoff,"oy")
plt.show()


#---------------------------------------------------------------------------------------------------------------
#   TOOL 2: CALCULATE THE SHORTEST PATH ONLY FOR THE Q347 OUTFLOW SO THAT THE STOPCONDITION SHORTEST PATH = TRUE
#---------------------------------------------------------------------------------------------------------------
## Make Loop which contains only the Timestep with the Q347 and then change the depth so that the stopcondition shortest path = True is satisfied.

## The loop that automates the function is still missing but it works manually!!!
## Timestep whith the Q347
shortestpathlistQ347key = []
shortestpathlistQ347value = []

q347 = Depth_TS["TS11"]

## Q347 only with dry nodes
q347array = np.asarray(q347)
q347array_transpose = q347array.transpose()

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

depthlist= []
for i in frange(0.3,0.1,0.01):
    depthlist.append(i)

i=0
while i <= 0.3:
    water_pointsq347=np.where(q347array_transpose<=i) #change this value 0.08 until the funcion nx.has_path is True
    water_pointsq347_array=np.asarray(water_pointsq347)
    water_pointsq347_array_transpose=water_pointsq347_array.transpose()

    ## matrix
    q347list = water_pointsq347_array.tolist()
    ## list
    q347nodes = q347list[0]
    ## Increase each number in the list by one because the nodes begin with 1 instead of 0
    q347nodes = [int(n+1) for n in q347nodes]
    q347nodes

    ## Find all edges incident on the nodes smaller than 0.2m - node pairs
    q347edges = G.edges(q347nodes)

    Q = nx.Graph()
    Q.add_nodes_from(q347nodes)
    Q.add_edges_from(q347edges)

    ## The nodes smaller than 0.2m must removed from the Graph to get the right graph
    QS = nx.Graph(G)
    QS.remove_nodes_from(q347nodes)
    QS.number_of_nodes()
    QS.number_of_edges()

    ## search 2 nodes which are in the graph
    search2nodesQ = QS.nodes
    source = min(search2nodesQ)
    target = max(search2nodesQ)

    ## Find shortest path
    nx.draw(QS,emd,node_color='k')
    plt.axis('equal')
    nx.has_path(QS,source,target)
    shortestpathQ = nx.has_path(QS,source,target)

    ## draw path in blue
    #path = nx.shortest_path(QS,source,target)
    #path_edges = zip(path,path[1:])
    #nx.draw_networkx_nodes(QS,emd,nodelist=path,node_color='b')
    #nx.draw_networkx_edges(QS,emd,edgelist=path_edges,edge_color='b',width=10)
    #plt.axis('equal')
    #plt.show()

    ## List with all timesteps shortestpath True and False
    shortestpathlistQ = [shortestpathQ]
    for element in shortestpathlistQ:
        if element is True:
            element=1
            waterdepthbig = i
            shortestpathlistQ347key.append(element)
            shortestpathlistQ347value.append(waterdepthbig)
        else:
            element=0
            waterdepthsmall = i
            shortestpathlistQ347key.append(element)
            shortestpathlistQ347value.append(waterdepthsmall)

    i=i+0.01
print "loop 2 done"

shortestpathlistQ347key.sort(reverse=True)
shortestpathlistQ347value.sort()

plt.title('Kuerzester Weg fuer das Q347')
plt.xlabel('Wassertiefe [cm]')
plt.ylabel('Durchgaengikeit')
plt.ylim(-0.05,1.05)
plt.grid(True)
plt.plot(shortestpathlistQ347value,shortestpathlistQ347key,shortestpathlistQ347value,shortestpathlistQ347key,"oy")
plt.show()


#----------------------------------------------------------------------------------------------
#   CSV EXPORT THE WATER DEPTH NODES FOR EVERY TIMESTEP
#----------------------------------------------------------------------------------------------
## Converting list with water depths to array to be able to export as csv and visualise the table in excel or join in Arc Gis
depthvaluesarray = np.asarray(depthvalues_float)
print depthvaluesarray.shape #first element is number of rows, second element number of columns
x,y = depthvaluesarray.shape
indices=np.tile(np.arange(1, y+1), (x, 1))
result=np.dstack((depthvaluesarray , indices)).astype(float, int)
depthvaluesarray_transpose=depthvaluesarray.transpose()

## Exporting array with the water depth for every node to csv
df = pd.DataFrame(depthvaluesarray_transpose)
df.to_csv(depthvaluesarray_transpose_csv)

Depth_TS_copy=copy.deepcopy(Depth_TS)
Node_ID=[]
for i in range (1,4086):
    Node_ID.append(i)

Depth_TS_copy['Node_ID']= Node_ID
print (len(Depth_TS_copy))
df_Depth_TS=pd.DataFrame(Depth_TS_copy)
df_Depth_TS.to_csv(depthvaluesarray_transpose_nodes_csv)
