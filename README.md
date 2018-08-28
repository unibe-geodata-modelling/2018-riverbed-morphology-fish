# 2018-riverbed-morphology-fish

------------------------------------------------------------
TOOLS TO ANALYZE THE RIVERBED MORPHOLOGY FOR THE FISHPATENCY
------------------------------------------------------------



INTRO
-----------------------------------------------------------------------------------
There is one script to analyze the results from the Guerbe River low Water simulation. 11 different scenarios with different outflows have been tested, 
of which each one returned an other water depth distribution. The different water depths of different discharges will be analyzed in terms of fish patency in the following script.


VARIABLES "basepath", "lines", "nodeshapefile", "depthsol"
-----------------------------------------------------------------------------------
For the river analysis you have to adjust the variable "basepath", (line 25) to your chosen workspace.

The "lines", variable (line 26, 27) is where you read in your edge data. ou have to adjust the path to the directory where you have stored the .edg file which comes from the program BASEMENT.

The "nodeshapefile", variable (line 28) is where you read in your node shapefile. There you have to adjust the path to the directory where your
node shapefile is stored. The node shapefile can be exported from QGIS and contains the Node_ID and the coordinates X, Y, Z. 

The "depthsol" variable (line 29) is where you read in the calculated water depth for each node and several time steps. You have to adjust the path to the directory where you have stored
the .sol file which comes from the program BASEMENT.


VARIABLES "depthvaluesarray_transpose_nodes_csv", "depthvaluesarray_transpose_csv"
-----------------------------------------------------------------------------------
In order to export the results in csv files the memory folder and the csv filename have to be defined.


TOOL DESCRIPTION
-----------------------------------------------------------------------------------
+ First, a graph is created (line 94 - 172)

+ Thereafter, the attribute water depth is added to the nodes (line 175 - 225)

+ Tool 1: Calculate the shortest path for different outflows (line 228 - 305)
In the first option, the outflow is changed so that the patency depth for a desired water depth (for example, 20 centimeters) is guaranteed.

+ Tool 2: Calculate the shortest path only for the Q347 outflow so that the stopcondition shortest path = True is satisfied. (line 308 - 363)
In the second option, the patency depth is reduced (for example 8 centimeters) to see from which water depth with the same outflow the patency is given.

+ At the end the reslutates are saved in a csv file to visualize them or to join in the Arc Gis (line 365 - 388)
