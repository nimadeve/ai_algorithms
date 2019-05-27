
from numpy import *  
import time  
import matplotlib.pyplot as plt 
from library import KMeans
class kmean:
	def __init__(self):
		## step 1: load data  
		print ("step 1: load data..." ) 
		dataSet = []
		fileIn = open("./data/kmean.csv")  
		for line in fileIn.readlines(): 
			temp=[]
			lineArr = line.strip().split('\t') 
			temp.append(float(lineArr[0]))
			temp.append(float(lineArr[1]))
			dataSet.append(temp)
		    
		fileIn.close()  
		## step 2: clustering...  
		print ("step 2: clustering..."  )
		dataSet = mat(dataSet)  
		k = 4  
		centroids, clusterAssment = KMeans.kmeans(dataSet, k) 
		  
		## step 3: show the result  
		print ("step 3: show the result..."  )
		KMeans.showCluster(dataSet, k, centroids, clusterAssment)