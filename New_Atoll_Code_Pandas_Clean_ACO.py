'''
Atoll Morphometric Code
'''
#%%---------------------Import python libaries-----------------------
import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from skimage import morphology
import pandas as pd
from skimage.measure import label, regionprops, find_contours#, points_in_poly, regionprops_table
import shutil
from shapely.geometry import Polygon, LineString
import cv2
#%%------------------input data
atollCompLoc = 'G:\Shared drives\Ortiz Atolls Database\CompositesWithCount\AllAtollsNew' #Location of all Atoll Composites (currently the ones made in 2019)
atollComp = os.listdir(atollCompLoc)
MH = [] # Marshall Islands 20/30 with 1 composite Millie to relook at       
PF = [] # French Polynesia 60/81       
TV = [] # Tuvala 5/7      
KI = [] # kiribati 10/20  
CR = [] # Caroline Islands 8/17
FJ = [] # Fiji 3/42  -  16 have large islands in the lagoon (volcanic) , 19 overclassify reef as motu (no motu actually present is some cases)
ID = [] # Indonesia 16/55
PG = [] # Papua New Guinea 12/26
GB = [] # United Kingdom 5/16 
CN = [] # China 0/6
A_all = [] # atolls in the Atlantic 0/18
P_all = [] # rest of the atolls in the Pacific 7/13
MV = [] # Republic of Maldives 5/25
IN = [] # india 3/13
MU = [] # Mauritius 0/7
SC = [] # Seychelles 1/5
I_all = [] # rest of the atolls in the Indian Ocean
for i in range(len(atollComp)):
    if atollComp[i][0:4] == 'P_MH':
        MH.append(atollComp[i])
    elif atollComp[i][0:4] == 'P_PF':
        PF.append(atollComp[i])
    elif atollComp[i][0:4] == 'P_TV':
        TV.append(atollComp[i])  
    elif atollComp[i][0:4] == 'P_KI':
        KI.append(atollComp[i])  
    elif atollComp[i][0:4] == 'P_CR':
        CR.append(atollComp[i])
    elif atollComp[i][0:4] == 'P_FJ':
        FJ.append(atollComp[i])  
    elif atollComp[i][0:4] == 'P_ID':
        ID.append(atollComp[i])  
    elif atollComp[i][0:4] == 'P_PG':
        PG.append(atollComp[i])  
    elif atollComp[i][2:4] == 'GB':
        GB.append(atollComp[i])  
    elif atollComp[i][0:4] == 'P_CN':
        CN.append(atollComp[i]) 
    elif atollComp[i][0:1] == 'A':
        A_all.append(atollComp[i]) 
    elif atollComp[i][0:1] == 'P':
        P_all.append(atollComp[i]) 
    elif atollComp[i][0:4] == 'I_MV':
        MV.append(atollComp[i]) 
    elif atollComp[i][0:4] == 'I_IN':
        IN.append(atollComp[i]) 
    elif atollComp[i][0:4] == 'I_MU':
        MU.append(atollComp[i]) 
    elif atollComp[i][0:4] == 'I_SC':
        SC.append(atollComp[i]) 
    else:
        I_all.append(atollComp[i])
issueLength = []
issueLengthIndex = [59,122,143,115,269,303,165,199,276,348,375,181,265,272]
for i in issueLengthIndex:
    issueLength.append(atollComp[i]) 
issueWidth = []
issueWidthIndex = [113,106,107,310,29,228,256,280,291,292,309,264,185,194,231,246,154,60,53]
for i in issueWidthIndex:
    issueWidth.append(atollComp[i]) 
issueLagoon = []
issueLagoonIndex = [65,72,22,322,324,351,366,380,270,342,361,207]
for i in issueLagoonIndex:
    issueLagoon.append(atollComp[i])    
   
#FileName = 'P_PF_Katiu_Atoll50c50mCountClip2.tif'
fileName = 'P_MH_Knox_Atoll50c50mCountClip2.tif'

atollName = fileName[0:-20]
print(atollName)

resolution = 30

morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
countryName = 'AllAtollsNew'
atollComp = os.listdir(atollCompLoc)

plt.close('all')
#%%------------------Set up the Functions
#Read in composite & matrix
def compositeReader(comp_filename):
    '''Reads in a Geotiff composite created in GEE and creates a matrix for all of the bands of the composite
    Inputs: comp_filename - full path to the composite
    Outputs: all 8 bands (blue, green, red, nir, swir1, swir2, count, mask)'''
    datafile = None
    datafile = gdal.Open(comp_filename)
    # creating an array for each band of the composite
    blue = datafile.GetRasterBand(1).ReadAsArray()
    green = datafile.GetRasterBand(2).ReadAsArray()
    red = datafile.GetRasterBand(3).ReadAsArray()
    nir = datafile.GetRasterBand(4).ReadAsArray()
    swir1 = datafile.GetRasterBand(5).ReadAsArray()
    swir2 = datafile.GetRasterBand(7).ReadAsArray()
    count  = datafile.GetRasterBand(8).ReadAsArray()
    mask = datafile.GetRasterBand(9).ReadAsArray()
    NDVI = (nir - red)/(nir + red)
    NDSI = (green - swir1)/(green + swir1)
    MNDWI = (green - swir2)/(green + swir2)
    #datafile = None
    return(blue, green, red, nir, swir1, swir2, count, mask, NDVI, MNDWI, NDSI,datafile)
    
#use K-means to classify the atoll into land, reef flat and water
def kMeans3(blue, green, red, nir, swir1, swir2, NDVI, MNDWI, NDSI):
    '''separates image into 3 groups using k-mean
    Inputs: bands
    Outputs: Lclassified array with 3 groups'''
    #find current size and shape of a band of the image
    l = np.size(blue)
    l2 = np.size(blue,0)
    l3 = np.size(blue,1)
    #reshape into a 1-D array (need in this format in order to run K-means)
    bluel = np.reshape(blue,(l)) 
    greenl = np.reshape(green,(l))
    redl = np.reshape(red,(l))
    nirl = np.reshape(nir,(l))
    swir1l = np.reshape(swir1,(l))
    swir2l = np.reshape(swir2,(l))
    
    #stack the 1-D array bands that will be used to run k-means
    pic1 = np.rot90(np.stack((bluel, greenl, redl, nirl, swir1l, swir2l)))
    # run k-means
    kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(pic1)
    #reshape into the original shape 
    L = np.rot90(np.rot90(np.reshape(kmeans,(l2,l3))))  
    return(L, kmeans)

def classifiedClean(L, kmeans, mask): 
    '''Ensures that Water = 0, Reef flat = 1, and Land = 2 by applying the following assumptions:
    water area > reef flat area and land area 
    total reef flat perimeter > total land perimeter
    Inputs L, kmeans (2D and 1D arrays of the k-means), mask band
    Outputs: Lclassified, Lclean8, Lclean64'''
    # calculate area of array pixels in L that are currently in 0, 1, and 2 and make the largest (one with the most pixels water)
    area0 = (kmeans == 0).sum()
    area1 = (kmeans == 1).sum()
    area2 = (kmeans == 2).sum()
    areas = (area0, area1, area2)
    maxAreaIndex = areas.index(max(areas)) # find the current number representing the max area (assumed to be water for atolls)
    minAreaIndex = areas.index(min(areas)) # find the current number representing the max area
    middleAreaIndex = 3 - maxAreaIndex - minAreaIndex # finds the remaining index
    k1 = np.copy(L)
    k1[L==minAreaIndex] = 1
    k1[L!=minAreaIndex] = 0
    k2 = np.copy(L)
    k2[L==middleAreaIndex] = 1
    k2[L!=middleAreaIndex] = 0
    
    label_k1, numk1 = label(k1,8,1,1)
    props = regionprops(label_k1)
    k1_perim = 0
    for i in range(numk1):
        k1_perim = k1_perim + props[i].perimeter
    
    label_k2, numk2 = label(k2,8,1,1)
    props = regionprops(label_k2)
    k2_perim = 0
    for i in range(numk2):
        k2_perim = k2_perim + props[i].perimeter
        
    if k1_perim > k2_perim: ##assume reef flat perimeter > land perimeter
        land = k2
    else:
        land = k1
    
    water = np.copy(L)
    water[L==maxAreaIndex] = 1
    water[L!=maxAreaIndex] = 0       
            
    water[np.isnan(mask)] = 1 #removes edge issues
    land[np.isnan(mask)] = 0 #removes edge issues
    reef = 1 - water
    
    Lclassified = np.copy(reef)
    Lclassified[land==1] = 2 
    
    Lclean8, intReef8 = cleanArray(land, reef, 8)
    Lclean64, intReef64 = cleanArray(land, reef, 64)
    return(Lclassified, Lclean8, Lclean64, intReef64)
    

#% Clean up land/reef flat   
def cleanArray(land, reef, smallestGroup):
    ''' removes small groups of pixels from reef flat, water, and motu with the groups size cutoff
    Inputs: binary reef, binary land, smallest size of pixal group to keep
    Outputs: cleaned image with water as 0, reef flat as 1, and motu as 2'''
    bLand = np.copy(land).astype(bool) # has to be a a boolean type to run next step
    binaryMotuLarge = morphology.remove_small_objects(bLand, smallestGroup, 1) #remove small land pixals
    binaryMotuLarge2 = morphology.remove_small_objects((1 - binaryMotuLarge.astype(int)).astype(bool), smallestGroup, 1) #remove small water pixals
    intMotu = 1 - binaryMotuLarge2.astype(int) #motu as 1 and everything else as zeros
    
    bReef = np.copy(reef).astype(bool) # has to be a a boolean type to run next step
    binaryReefLarge = morphology.remove_small_objects(bReef, smallestGroup, 1) #remove small land pixals
    binaryReefLarge2 = morphology.remove_small_objects((1 - binaryReefLarge.astype(int)).astype(bool), smallestGroup, 1)
    intReef = 1 - binaryReefLarge2.astype(int) #reef as 1 and everything else as zeros
    
    Lclean = np.copy(intReef)
    Lclean[intMotu==1] = 2
    return(Lclean, intReef)
    
# clean binary 
def cleanbinary(land, smallestGroup):
    ''' removes small groups of pixels from the groups size cutoff
    Inputs: binary ing, and the smallest size of pixal group to keep
    Outputs: cleaned image'''
    bLand = np.copy(land).astype(bool) # has to be a a boolean type to run next step
    binaryMotuLarge = morphology.remove_small_objects(bLand, smallestGroup, 1) #remove small land pixals
    intMotu = binaryMotuLarge.astype(int) #motu as 1 and everything else as zeros    
    return intMotu

#Identify every object    
def identifyRegion(binaryImg):
    '''Given a binary image this function this function runs regionprops  
    Label regions and then - Region-props of given input binary image
    Inputs: binary image of interested objects (lagoon, reef-flat, or motu)
    Outputs: index of objects, area perimeter, centroid'''
    label_binaryImg, num_binaryImg = label(binaryImg,8,1,1)
    props_binaryImg = regionprops(label_binaryImg)
    return(label_binaryImg, num_binaryImg, props_binaryImg)


#------------------Lagoon identifying code
def UserNumLagoons(img):
    plt.figure(1)
    plt.imshow(img) 
    plt.pause(0.1)# work around to actually get the image to display before it asks for user input
    plt.show()       
    UserNumL = int(input("Enter number of Lagoons: "))
    
    print("Please click the centerpoint of the " + str(UserNumL) + " lagoons in order from largest to smallest.")
    x = plt.ginput(UserNumL)
    plt.close()
    plt.pause(0.1)
    x = [[int(j) for j in i] for i in x] #remember ginput gives x[0] --> use as x value to plot, x[1] --> use as y value to plot
    #but remember when using in array, use x[1] as ROW (first input), x[0] as COLUMN, second input
    return(UserNumL, x)

def CheckNumLagoons(img, UserNumL, cp):
    plt.figure(1)
    plt.imshow(img)
    plt.pause(0.1)
    plt.show()
    x_val = [x[0] for x in cp]
    y_val = [x[1] for x in cp]
    plt.plot(x_val,y_val,'or')
    plt.pause(0.1)# work around to actually get the image to display before it asks for user input
     
    correctL = input("Are there " + str(UserNumL) + " lagoons with the centerpoints correctly identified (y/n): ")
    plt.close()
    plt.pause(0.1)
    return(correctL)

def stringToPairList(pointStr,num):    
    centerPoints3 = pointStr.split(',')    
    centerPoints5 = []
    
    for i in centerPoints3:
        k = int(''.join(c for c in i if c.isdigit()))
        centerPoints5.append(k)    
    centerPoints = []
    for i in range(0,num*2,2):
        cp = [centerPoints5[i],centerPoints5[i+1]] 
        centerPoints.append(cp)
    return centerPoints

#------------------ more lagoon code
def ErosionDilation(img, kernalNum):
    ''' performs erosion and dilation on a binary image and then removes small water and land bodies that are not the lagoons (by size)
    Inputs: Binary image, size of kernal to perform the erosion and dilation with
    Outputs: The processed Image'''
    kernel = np.ones((kernalNum,kernalNum))
    erosion = cv2.erode((img*1.0).astype(np.float32),kernel,iterations=1)
    dilate = cv2.dilate((erosion*1.0).astype(np.float32),kernel,iterations=1) 
    water2 = dilate.astype(int)
    return(water2)

def OpeningClosing(img, kernalNum):
    ''' performs erosion and dilation on a binary image and then removes small water and land bodies that are not the lagoons (by size)
    Inputs: Binary image, size of kernal to perform the erosion and dilation with
    Outputs: The processed Image'''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalNum,kernalNum))
    # closing = cv2.morphologyEx((1-img*1.0).astype(np.float32), cv2.MORPH_CLOSE,kernel,iterations=1)
    # opening = cv2.morphologyEx((closing*1.0).astype(np.float32), cv2.MORPH_OPEN ,kernel,iterations=1)
    closing = cv2.morphologyEx((1-img*1.0).astype(np.float32), cv2.MORPH_CLOSE,kernel,iterations=1)
    water2 = 1-closing.astype(int)
    return(water2)
    
def autoLagoonFinder(intWater, numLagoon, cp):
    '''finds all the lagoons for the given atoll image
    Inputs: intwater and number of lagoons, and center points of lagoons
    outputs: label image, # lagoons found, which ones found based on center points'''     
    waters = cleanbinary(intWater, 256)
    label_water_img, numWater = label(waters,8,1,1)
    kernalList = np.array((1, 3, 5, 7))
    i = 0
    
    while numWater != (numLagoon + 1) and i < 4:
        water1 = OpeningClosing(waters, kernalList[i])
        waters = cleanbinary(water1, 256)
        waters[waters - intWater < 0] = 0
        label_water_img, numWater, props_water = identifyRegion(waters)
        i = i + 1
    
    label_water_img_bySize = labelImgBySize(waters)
    cpTF = []
    for i in cp:
        j = label_water_img_bySize[int(i[1]), int(i[0])] #this is correct, i[1] first, i[0] second for ginput values
        cpTF.append(j != 0)
    return(waters, label_water_img_bySize, numWater, cpTF)

def labelImgBySize(bImg):
    '''Takes in a binary image and identifys the pieces by size using region props
    Inputs: a binary image
    Outputs: a labeled by size image'''
    label_img, num, props = identifyRegion(bImg)
    label_img_size = np.copy(label_img)
    areas = np.zeros((num))
    for i in range(num):
        areas[i] = props[i].area
    for i in range(num):
        label_img_size[label_img == np.argmax(areas) + 1] = i + 1
        areas[np.argmax(areas)] = 0
    label_img_size[label_img == label_img[0,0]] = 0
    return(label_img_size)

def manualLagoonFinder(intWater, numLagoon, cp):
    #change to always add the lines to intwater
    waters, label_water_img_bySize, numWater, cpTF = autoLagoonFinder(intWater, numLagoon, cp)
    water1 = np.copy(intWater)
    
    f= open("lagoon_closer.txt","a+")
    f.seek(0, 0)
    
    n = 1
    while n < 70 and not (all(cpTF) == True):
        line = f.readline()
        if line != '':
            x = stringToPairList(line,2)
        else:
            x = getPoints(waters, label_water_img_bySize)
            f.write(str(x) + "\n")
        
        dist = ((x[0][1] - x[1][1])**2 + (x[0][0] - x[1][0])**2)**0.5
        for k in range(int(dist)):    
            spot1 = int(((x[0][1] - x[1][1])/int(dist))*k + x[1][1])
            spot2 = int(((x[0][0] - x[1][0])/int(dist))*k + x[1][0])
            water1[spot1,spot2] = 0
            water1[spot1+1,spot2] = 0
            water1[spot1,spot2+1] = 0
            water1[spot1+1,spot2+1] = 0
   
        waters, label_water_img_bySize, numWater, cpTF = autoLagoonFinder(water1, numLagoon, cp)
        print(numWater, cpTF, all(cpTF) == True)
        n = n + 1
    f.close()
    return(waters, label_water_img_bySize, numWater, cpTF)

def getPoints(img, displayImg):                   
    plt.figure(1)
    plt.imshow(displayImg) 
    plt.pause(0.1)# work around to actually get the image to display before it asks for user input
    
    print("Please click two points to close a gap in the lagoon.")
    x = plt.ginput(2)
    x = [[int(j) for j in i] for i in x]
    plt.close()
    plt.pause(0.1)    
    return(x) 

def perim_object(label_img, ind):
    '''get the perimeter points'''
    OneMotu64 = np.zeros_like(label_img)
    OneMotu64[label_img==ind+1] = 1 
    OnePerimM = find_contours(OneMotu64)
    return(OnePerimM) 

#----------------------Functions needed to run the ocean/lagoon point code
def closestPoint(onePoint, pointList):
    ''' This function finds the closest point and returns that distance
    
    Parameters
    ----------
    onePoint : contains the x and y parameters of the point that we want to find the closest point too
    pointList : contains a list of points to chek in and find the closest one
    
    Returns
    -------
    distmin : the minimum distance fron the one point to any other point in pointList
    '''
    dist = ((onePoint[1] - pointList[:,0])**2 + ((onePoint[0] - pointList[:,1])**2))**0.5
    distmin = np.min(dist)
    return distmin

def closestPointExpand(onePoint, pointList):
    ''' This function finds the closest point and returns that distance, and the coordinates of that point
    Parameters
    ----------
    onePoint : contains the x and y parameters of the point that we want to find the closest point too
    pointList : contains a list of points to chek in and find the closest one
    
    Returns
    -------
    distmin : the minimum distance fron the one point to any other point in pointList
    pointLoc_x : x coordinate of the closest point
    pointLoc_y : y coordinate of the closest point
    '''
    dist = ((onePoint[1] - pointList[:,0])**2 + ((onePoint[0] - pointList[:,1])**2))**0.5
    ang = ((90 - np.arctan2((onePoint[1] - pointList[:,0]),(pointList[:,1] - onePoint[0]))*180/np.pi)+360) % 360
    ang[dist == 0] = np.nan
    distmin = np.min(dist)
    pointLoc_x = pointList[np.argmin(dist),1]
    pointLoc_y = pointList[np.argmin(dist),0]
    point_ang = ang[np.argmin(dist)]
    
    return distmin, pointLoc_x, pointLoc_y, point_ang

def crossCheck(objPoint, polygonObj):
    ''' checks if the line fron the perimeter point to the closest point found in the closestPointExpand crosses the polygon.
    checks this only for the ocean points becasuse the lagoon points are more likey to need to cross the polygon regardless
    Parameters
    ----------
    objPoint : contains the x and y coords for both ends of the line and a string if it is currently called an ocean or a lagoon point
    polygonObj : a shaply polygon of the object 
    Returns
    -------
    1 if it needs to stay a lagoon point or change to a lagoon point and 0 if it should stay an ocean point
    '''
    polygonObj=polygonObj.buffer(0)#added in because of The operation 'GEOSTouches_r' could not be performed. Likely cause is invalidity of the geometry <shapely.geometry.polygon.Polygon object at 0x0000021B4D75AD00>
    if objPoint[4] == 'lagoon':
        return(1)
    line1 = LineString([(objPoint[0],objPoint[1]),(objPoint[2],objPoint[3])])
    if objPoint[0] == objPoint[2] and objPoint[1] == objPoint[3]:
        return(0)
    if line1.touches(polygonObj):
        return(0)
    return(1)
    
def PolygonPandas(x,y):
    ''' this function creates a shaply polygon from two x and y Pandas columns
    Inputs: x and y Pandas columns
    Outputs: po is the shaply polygon
    '''
    x1 = x.to_list()
    y1 = y.to_list()
    l = (list(pair) for pair in zip(x1, y1))
    po = Polygon(l)
    return(po)    
    
#-----------------ocean lagoon points
def OceanLagoonOuter(m, a, l, s1, s2):
    '''
    This function deternines if the perimeter points of an onject are to be classified as an ocean or a lagoon point
    Inputs: m is a subset of the main dataframe for one object (normaly one motu or reef flat) 
    a are the atolls ocean side points (all the outer points)
    l are the points a,ong the perimeter of all the lagoons
    s1 is the dataframe column name with the x coords of the points at will be assigned as ocean/lagoon
    s2 is the dataframe column name with the the y coords of the points at will be assigned as ocean/lagoon    
    Outputs: a 'label' column with ocean/lagoon points identified
    '''
    mt =m.copy()
    mt[['ocean distance', 'ocean x', 'ocean y', 'ocean angle']] = mt[[s1,s2]].apply(closestPointExpand, pointList = a, axis = 1,result_type ='expand')
    mt[['lagoon distance', 'lagoon x', 'lagoon y', 'lagoon angle']] = mt[[s1,s2]].apply(closestPointExpand, pointList = l, axis = 1,result_type ='expand')
    mt['ocean distance'] = mt['ocean distance'] - mt['ocean distance'].min()
    mt['lagoon distance'] = mt['lagoon distance'] - mt['lagoon distance'].min()
    mt['lagoon angle'][np.isnan(mt['lagoon angle'])] = (mt['ocean angle'] + 180)%360
    mt['dist check'] = mt['ocean distance'] -  mt['lagoon distance']
    
    mt['label'] = 'ocean'
    mt['label'][mt['dist check']>0] = 'lagoon'
    mt['label int'] = mt['label'].map({'ocean': 0, 'lagoon': 1})
    
    # only run the cross check for the motu not for the reef flat
    po = PolygonPandas(mt[s1],mt[s2])    # create shaply polygon
    mt['label int'] = mt[[s1,s2,'ocean x', 'ocean y','label']].apply(crossCheck, polygonObj = po, axis = 1,result_type ='expand')
    #filter line above to only ocean points for applying
    
    mt['diff t'] = mt['label int'].diff()
    
    #count = 1
    while np.nansum(abs(mt['diff t'])) > 2:
        ind = mt.loc[mt['diff t'] != 0].index
        ind2 = np.diff(ind)
        imin = np.argmin(ind2)
        mt.loc[ind[imin]:ind[imin+1]-1,'label int'] = 1 - mt.loc[ind[imin]:ind[imin+1]-1,'label int']
        mt['diff t'] = mt['label int'].diff()
        # print(count)
        # count = count + 1
        
    mt['label'] = mt['label int'].map({0: 'ocean', 1:'lagoon'})
    
    # if a motu is all lagoon or all ocean this code will fix it
    a2 = len(mt['label'].unique())
    if a2 == 1:
        ind = mt[['ocean distance']].idxmin()
        mt['label'].loc[ind] = 'ocean'
        ind = mt[['lagoon distance']].idxmin()
        mt['label'].loc[ind] = 'lagoon'
    
    mt['last point'] = 0
    mt['last point'].iloc[-1] = 1
    
    return mt[['label','lagoon angle', 'last point']]
    
def OceanLagoonReefOuter(m, a, l, s1, s2):
    '''
    This function deternines if the perimeter points of an onject are to be classified as an ocean or a lagoon point
    Inputs: m is a subset of the main dataframe for one object (normaly one motu or reef flat) 
    a are the atolls ocean side points (all the outer points)
    l are the points along the perimeter of all the lagoons
    s1 is the dataframe column name with the x coords of the points at will be assigned as ocean/lagoon
    s2 is the dataframe column name with the the y coords of the points at will be assigned as ocean/lagoon    
    Outputs: a 'label' column with ocean/lagoon points identified
    '''
    mt =m.copy()
    mt[['ocean distance', 'ocean x', 'ocean y', 'ocean angle']] = mt[[s1,s2]].apply(closestPointExpand, pointList = a, axis = 1,result_type ='expand')
    mt[['lagoon distance', 'lagoon x', 'lagoon y', 'lagoon angle']] = mt[[s1,s2]].apply(closestPointExpand, pointList = l, axis = 1,result_type ='expand')
    mt['lagoon angle'][np.isnan(mt['lagoon angle'])] = (mt['ocean angle'] + 180)%360
    mt['dist check'] = mt['ocean distance'] -  mt['lagoon distance']
    
    mt['label'] = 'ocean'
    mt['label'][mt['dist check']>0] = 'lagoon'
    mt['label int'] = mt['label'].map({'ocean': 0, 'lagoon': 1})
    
    # mt2 = mt.copy()
    # mt2.sort_values(['label int', 'order'], axis = 0)
    
    mt['last point'] = 0
    mt['last point'].iloc[-1] = 1
    
    # if a reef flat is all lagoon or all ocean this code will  fix it
    a2 = len(mt['label'].unique())
    if a2 == 1:
        ind = mt[['ocean distance']].idxmin()
        mt['label'].loc[ind] = 'ocean'
        ind = mt[['lagoon distance']].idxmin()
        mt['label'].loc[ind] = 'lagoon'
    return mt[['label','lagoon angle','last point']]      
             
def orderPoints(m):
    mt = m.copy()    
    mt['label int'] = mt['o/l label'].map({'ocean': 0, 'lagoon': 1})
    mt['diff'] = mt['label int'].diff()
    numS = mt['diff'].abs().sum()
    if numS == 2:
        k2 = mt[mt['diff'].abs() == 1].index
        mt2 = mt.loc[k2[0]:mt.tail(1).index[0]]
        mt2 = mt2.append(mt.loc[mt.head(1).index[0]:k2[0]-1])
        mt = mt2
    return mt.drop(['diff', 'label int'], axis = 1)

def shoreNormalAngle(m, s1, s2, s3):
    '''create the angle normal to the shoreline that points towards the lagoon'''
    mt =m.copy()
    mt['x diff'] = mt[s1].diff(periods = 2) - mt[s1].diff(periods = -2)
    mt['y diff'] = mt[s2].diff(periods = -2) - mt[s2].diff(periods = 2)
    mt['shore angle'] = (90 - np.arctan2(mt['y diff'] , mt['x diff'])*180/np.pi) % 360
    mt['shore normal'] = (mt['shore angle'] + 90) % 360
    mt['angle diff'] = (mt[s3] - mt['shore angle']) % 360
    mt['shore normal'][mt['angle diff']>180] = (mt['shore normal'] - 180)% 360
    mt['shore normal'] = mt['shore normal'].fillna(method = 'ffill').fillna(method = 'bfill')
    return mt['shore normal']

#---------------Width code
def cp2(op, pl1, pl2, angT, distT, r):
    '''cp2 takes in a point, a list of target points and the shore normal angle of the single point
    it returns the shortest near normal distance and the coords of that point
    '''
    dist = (((op[0] - pl1[:])**2 + ((op[1] - pl2[:])**2))**0.5)*r
    dist[dist==0]=np.nan
    closeDistmin = np.min(dist)
    #print(closeDistmin, op, len(pl1), np.argmin(dist))
    pointLoc_x = pl1.iloc[np.argmin(dist)]
    pointLoc_y = pl2.iloc[np.argmin(dist)]
    
    ang = ((90 - np.arctan2((op[1]-pl2[:]),(pl1[:] - op[0]))*180/np.pi) + 360) % 360
    #angR = ang.iloc[np.argmin(dist)]
    
    target = op[2]
    ang2 = (ang - target)%360
    ang2[ang2>180] = ang2 - 360
    ang3 = abs(ang2)
    dist[ang3>angT] = np.nan
    ang4 = ang3.sort_values()
    distmin = np.min(dist[ang4.index])
    if distmin/closeDistmin < distT:
        pointLoc_x = pl1[ang4.index].iloc[np.argmin(dist[ang4.index])]
        pointLoc_y = pl2[ang4.index].iloc[np.argmin(dist[ang4.index])]
        #angR = ang[ang4.index].iloc[np.argmin(dist[ang4.index])]
    else:
        distmin = closeDistmin
    return distmin, pointLoc_x, pointLoc_y#, angR

def WidthOuter(m, m2, s1, s2, s3, s4, s5, s6, res, angTarget = 15, distTarget = 2):
    '''
    This code calculates the width
    '''
    mt =m.copy()
    mt2 = m2.copy()
    mt3 = mt2[mt2[s6] == mt[s6].iloc[0]]
    m3 = mt[[s1,s2,s5]].apply(cp2, pl1 = mt3[s3], pl2 = mt3[s4], angT = angTarget, distT = distTarget, r = res, axis = 1,result_type ='expand')
    return m3

def WidthOuterR(m, m2, mo, s1, s2, s3, s4, s5, s6, s7, s8, res, angTarget = 15, distTarget = 2):
    '''
    This code calculates the width
    '''
    mt =m.copy()
    mt2 = m2.copy()
    mo2 = mo.copy()
    mt3 = mt2[mt2[s6] == mt[s6].iloc[0]]
    mo3 = mo2[mo2[s6] == mt[s6].iloc[0]]
    mt4 = mt3.append(mo3)
    mt4['x'] = mt4[s3]
    k = mt4['x'].isna()
    mt4.loc[k.values,'x'] =mt4.loc[k.values,s7]
    mt4['y'] = mt4[s4]
    mt4.loc[k.values,'y'] =mt4.loc[k.values,s8]
    m3 = mt[[s1,s2,s5]].apply(cp2, pl1 = mt4['x'], pl2 = mt4['y'], angT = angTarget, distT = distTarget, r = res, axis = 1,result_type ='expand')
    return m3


#---------------- length code functions
def LengthInner(xy, r):
    dist = ((xy[1:,1]-xy[:-1,1])**2 + (xy[1:,0]-xy[:-1,0])**2)**0.5*r
    length = dist.sum()
    return length

def LengthInnerSkip(xy, r, skip):
    dist = ((xy[1:,1]-xy[:-1,1])**2 + (xy[1:,0]-xy[:-1,0])**2)**0.5
    dist[dist > skip] = np.nan
    length = np.nansum(dist)*r
    return length

def LengthSort(xy): # add nearest non zero distance
    line2 = xy[0,:]
    intpoint = 0
    
    while(intpoint < len(xy)-1):
        dist = (((xy[intpoint,0] - xy[:,0])**2 + ((xy[intpoint,1] - xy[:,1])**2))**0.5)
        dist[:intpoint+1] = np.nan
        dist[dist == 0] = np.nan
        #np.nanmin(dist)
        intpoint =  len(dist) - 1 - np.nanargmin(dist[::-1])
        line2 = np.vstack((line2, xy[intpoint,:]))
    return line2   

def LengthSortHyp(xy, r): # add nearest non zero distance
    line2 = xy[0,:]
    intpoint = 0
    
    while(intpoint < len(xy)-1):
        dist = (((xy[intpoint,0] - xy[:,0])**2 + ((xy[intpoint,1] - xy[:,1])**2))**0.5)
        dist[:intpoint+1] = np.nan
        dist[dist == 0] = np.nan
        dist2 = dist.copy()
        dist2[dist > r*r**0.5] = np.nan
        dist2[1] = 0
        intpoint = np.max(((len(dist) - 1 - np.nanargmin(dist[::-1])), (len(dist2) - 1 - np.nanargmax(dist2[::-1]))))
        line2 = np.vstack((line2, xy[intpoint,:]))
    return line2   
    
def LengthOuter(m, s1, s2, s3, s4, res, rou = 1):
    '''
    This code calculates the length
    rou is the rounding in number of pixals
    '''
    mt =m.copy()
    mt['center x'] = (mt[s1] + mt[s3])/2
    mt['center y'] = (mt[s2] + mt[s4])/2
    
    xy = mt[['center x','center y']].values
    xyr = rou *np.around(xy/rou)
    
    indexes = np.unique(xyr, axis = 0, return_index=True)[1]
    xyrs = np.array([xyr[index] for index in sorted(indexes)])
    
    oceanxy = mt[[s1,s2]].values
    xy2 = np.vstack((oceanxy[0,:], xyrs, oceanxy[-1,:]))
    line2 = LengthSortHyp(xy2,rou)[::-1,:]
    line3 = LengthSortHyp(line2,rou)[::-1,:]
    length = LengthInner(line3, res)  
    
    mt['length'] = length
    
    # plt.plot(line3[:,0],line3[:,1],c = 'k')#, s = 2)
    # plt.scatter(line3[:,0],line3[:,1], c = 'k')
    
    return mt['length']

def LengthOuterSimple(m, s1, s2, res):
    '''
    This code calculates the length of a list of points from a pandas dataframe
    '''
    mt =m.copy()
    length = LengthInner(mt[[s1,s2]].values, res)      
    mt['length'] = length
    
    return mt['length']

def LengthOuterReef(m, s1, s2, res):
    '''
    This code calculates the length of a list of points from a pandas dataframe
    '''
    mt =m.copy()
    length = LengthInnerSkip(mt[[s1,s2]].values, res, 3)      
    mt['length'] = length
    
    return mt['length']

def write_geotiff(fn, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(fn, arr.shape[2], arr.shape[1], arr.shape[0], arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    blist = list(range(1,arr.shape[0]+1))
    out_ds.WriteRaster(0,0,arr.shape[2],arr.shape[1],arr.tostring(),arr.shape[2],arr.shape[1],band_list=blist)
    out_ds.FlushCache()

def addcrop(old, xpts):
    #takes in old array and x points from top left, bottom right ginput, returns new array cropped to that box
    new = old[xpts[0][1]:xpts[1][1],xpts[0][0]:xpts[1][0]]
    return new

def EfW2(op, avgWidth, mInd):
    if op[0] == 0:
        df3['motu present'].loc[op.name] = 2
        df3['motu index'].loc[op.name] = mInd
        df3['effective reef flat width'].loc[op.name] = avgWidth

def EfW2_outer(m, m2):
    mt2 = m2.copy()
    m4 = mt2[mt2['motu index'] == m.name]
    m5 = mt2.loc[m4.index[0]:m4.index[-1]]
    m5[['motu present','motu index','effective reef flat width']].apply(EfW2, avgWidth = m[1], mInd = m.name, axis = 1)

#calc. stoddart values
def calcStoddart(A1,A2,d1,L1,c,perim):
    #ok this function takes area of object (A1), area of equivalent circle with same perimeter (A2),
    #diameter of equivalent circle with same area (d1), major axis length (L1),
    #centroid and perimeter points
    #it will return the Horton Form factor (F),Millers circularity ratio (Rc),Schummans elongation (Re),
    #Ellipcity index (Ie),and radial line shape index (Ir) based on Stoddart 1965
    F = A1/L1**2
    Rc = A1/A2
    Re = d1/L1
    b = A1/(np.pi*(L1/2))
    Ie = L1/(2*b)
    return F, Rc, Re, Ie

# effective reef flat width ---- All this following code is to get the effective reef flat to work
#---------------Width code
def cp2Ef(op, pl1, pl2, angT, distT, r):
    '''cp2 takes in a point, a list of target points and the shore normal angle of the single point
    it returns the shortest near normal distance and the coords of that point
    '''
    dist = (((op[0] - pl1[:])**2 + ((op[1] - pl2[:])**2))**0.5)*r
    closeDistmin = np.min(dist)
    #print(closeDistmin, op, len(pl1), np.argmin(dist))
    pointLoc_x = pl1.iloc[np.argmin(dist)]
    pointLoc_y = pl2.iloc[np.argmin(dist)]
    pointLoc_ind = pl1.index[np.argmin(dist)]
    ang = ((90 - np.arctan2((op[1]-pl2[:]),(pl1[:] - op[0]))*180/np.pi) + 360) % 360
    angR = ang.iloc[np.argmin(dist)]
    target = op[2]
    ang2 = (ang - target)%360
    ang2[ang2>180] = ang2 - 360
    ang3 = abs(ang2)
    dist[ang3>angT] = np.nan
    ang4 = ang3.sort_values()
    distmin = np.min(dist[ang4.index])
    if distmin/closeDistmin < distT:
        pointLoc_x = pl1[ang4.index].iloc[np.argmin(dist[ang4.index])]
        pointLoc_y = pl2[ang4.index].iloc[np.argmin(dist[ang4.index])]
        angR = ang[ang4.index].iloc[np.argmin(dist[ang4.index])]
        pointLoc_ind = pl1[ang4.index].index[np.argmin(dist[ang4.index])]
    else:
        distmin = closeDistmin
    return distmin, pointLoc_x, pointLoc_y, pointLoc_ind#, angR

def WidthOuterEf(m, m2, s1, s2, s3, s4, s5, s6, res, angTarget = 15, distTarget = 2):
    '''
    This code calculates the width
    '''
    mt =m.copy()
    mt2 = m2.copy()
    mt3 = mt2[mt2[s6] == mt[s6].iloc[0]]
    m3 = mt[[s1,s2,s5]].apply(cp2Ef, pl1 = mt3[s3], pl2 = mt3[s4], angT = angTarget, distT = distTarget, r = res, axis = 1,result_type ='expand')
    return m3

def EfW(op):
    if not np.isnan(op[0]):
        df3['motu present'].loc[op[0]] = 1
        df3['motu index'].loc[op[0]] = op[1]
        df3['effective reef flat width'].loc[op[0]] = np.minimum(op[2], df3['effective reef flat width'].loc[op[0]])

def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)
#%% organize files
full_path = morphOutput + '\\' + countryName + '\\' + atollName # create county and atoll directory if they do not exist
if not os.path.exists(full_path):
    #%----------------------- create folder for the atoll if it does not exist--------------
    os.chdir(morphOutput)
    full_path = morphOutput + '\\' + countryName + '\\' + atollName # create county and atoll directory if they do not exist
    os.makedirs(full_path,exist_ok=True)
    os.chdir(full_path) # set working directory to the atoll directory
    if not os.path.isfile(fileName): # Copy the composite into its atoll folder if it is not already there
        src = atollCompLoc + '\\' + fileName
        dst = full_path
        shutil.copy(src, dst)
full_path = morphOutput + '\\' + countryName + '\\' + atollName # create county and atoll directory if they do not exist
os.chdir(full_path) # set working directory to the atoll directory        
full_path = atollCompLoc + '\\' + fileName # create county and atoll directory if they do not exist   
#%% Read in composite, add crop of image if necessary, save new image
blue, green, red, nir, swir1, swir2, count, mask, NDVI, MNDWI, NDSI,ds = compositeReader(full_path)
plt.figure(1)
plt.imshow(blue) 
plt.pause(0.1)# work around to actually get the image to display before it asks for user input
plt.show()
print("Please click two points to crop image top left, bottom right of area to remain")
x = plt.ginput(2)
x = [[int(j) for j in i] for i in x]
plt.close()
plt.pause(0.1) 

blue = addcrop(blue,x)
green = addcrop(green,x)
red = addcrop(red,x)
nir = addcrop(nir,x)
swir1= addcrop(swir1,x)
swir2= addcrop(swir2,x)
count= addcrop(count,x)
mask= addcrop(mask,x)
NDVI= addcrop(NDVI,x)
MNDWI= addcrop(MNDWI,x)
NDSI = addcrop(NDSI,x)

data = np.stack((blue, green, red, nir, swir1, swir2, count, mask),axis=2)
data2 = data.transpose((2,0,1))#convert to bands, rows, columns

fname = fileName[0:-4]+"_old.tif"
shutil.copyfile(full_path,fname) #save old composite
write_geotiff(fileName,data2,ds) #save new cropped composite

#need to also remove lagoon data if cropping the image
#if you crop, must delete lagoon helper files!!
if os.path.exists('lagoon_helper.txt'):
    full_path = morphOutput + '\\' + countryName + '\\' + atollName + "\\lagoon_helper.txt"
    fname = "lagoon_helper_old.txt"
    shutil.copyfile(full_path,fname)
    os.remove('lagoon_helper.txt')
if os.path.exists('lagoon_closer.txt'):
    full_path = morphOutput + '\\' + countryName + '\\' + atollName + "\\lagoon_closer.txt"
    fname = "lagoon_closer_old.txt"
    shutil.copyfile(full_path,fname)
    os.remove('lagoon_closer.txt')
#%% kmeans classification
L, kmeans = kMeans3(blue, green, red, nir, swir1, swir2, NDVI, MNDWI, NDSI)
#%% add mask
plt.figure(1)
plt.imshow(L) 
plt.pause(0.1)# work around to actually get the image to display before it asks for user input
plt.show()
print("Please click two points to mask something out top left, bottom right")
x = plt.ginput(2)
x = [[int(j) for j in i] for i in x]
plt.close()
plt.pause(0.1) 

mask[x[0][1]:x[1][1],x[0][0]:x[1][0]] = np.nan
#%% plot images of cleaned kmeans
Lclassified, Lclean8, Lclean64, intReef64 = classifiedClean(L, kmeans, mask)
#% plot figures
plt.figure(15)
plt.imshow(Lclean8)
plt.show()

plt.figure(14)
plt.imshow(L)
plt.show()

data = np.stack((NDVI,MNDWI,NDSI),axis = -1)
data = np.stack(((NDVI+1)/2,(MNDWI+1)/2,(NDSI+1)/2),axis = -1)
plt.figure(16)
plt.imshow(data)
plt.show()
data = np.stack((red,green,blue),axis = -1)
plt.figure(18)
plt.imshow(data)
plt.show()

plt.figure(17)
plt.imshow(count)
plt.colorbar()
plt.show()
plt.pause(10)
plt.close('all')
#%% save Images
data = np.stack((red,green,blue),axis = -1)
plt.imsave('RGB.png', data)
data = np.stack(((NDVI+1)/2,(MNDWI+1)/2,(NDSI+1)/2),axis = -1)
plt.imsave('IndexStack.png', data)
plt.imsave('Kmeans.png', L)
#%% number and center of lagoons from user or file
if not os.path.isfile('lagoon_helper.txt'):
    numL, centerPoints = UserNumLagoons(intReef64)
    f= open("lagoon_helper.txt","w+")
    f.write('UserNumL = ' + str(numL) + "\n")
    f.write(str(centerPoints))
    f.close()
    
if os.path.isfile('lagoon_helper.txt'):
    f= open("lagoon_helper.txt","r+")
    line = f.readline()
    numL = int(line[11])
    line = f.readline()
    centerPoints = stringToPairList(line,numL)
    f.close()
#%% trys to automatically find the lagoon
intWater64 = 1 - intReef64      
waters, label_water_img, numWater, centerPointsTF = autoLagoonFinder(intWater64, numL, centerPoints)   
#%% get points to close lagoon manually if needed
if all(centerPointsTF) == True and numL == numWater - 1:
    plt.imshow(label_water_img)
else:
    waters, label_water_img, numWater, centerPointsTF = manualLagoonFinder(intWater64, numL, centerPoints) 
    plt.imshow(label_water_img)

print(atollName)

#%% region props on the water

props_water = regionprops(label_water_img)
#% Create the list if the ocean and the lagoon points for atoll 
# remove inner reef flat points
# start by creating a list of ocean points and lagoon points from the atoll itself
waters2 = 1 - waters

label_waters2, num_waters2, props_waters2 = identifyRegion(waters2)

largestLagoon = label_waters2[centerPoints[0][1],centerPoints[0][0]]-1
oceanArea = props_waters2[label_waters2[0,0]-1].area
largestLagoonArea = props_waters2[largestLagoon].area
smallestLagoon = label_waters2[centerPoints[-1][1],centerPoints[-1][0]]-1
smallestLagoonArea = props_waters2[smallestLagoon].area
smallestArea = min(oceanArea,smallestLagoonArea)
reefArea = (Lclassified == 1).sum()

waters3 = 1 - cleanbinary(waters2, reefArea/2)
waters3 = cleanbinary(waters3, smallestArea-1)

label_waters3, num_waters3, props_waters3 = identifyRegion(waters3)

OnePerim = find_contours(waters3)

atollOceanPerim = OnePerim[0]
atollLagoonPerim = OnePerim[1]

for i in range(1,numL):
    atollLagoonPerim = np.append(atollLagoonPerim,OnePerim[i+1], axis = 0)

a = atollOceanPerim 
l = atollLagoonPerim

largeLagoonCentroid = props_waters3[label_waters3[centerPoints[0][1],centerPoints[0][0]]-1].centroid
largeLagoonCentroidInt = (int(largeLagoonCentroid[0]),int(largeLagoonCentroid[1]))
#% creates a binary images of the motu and removesinterior motu pieces

intMotu64 = np.copy(Lclean64)
intMotu64[intMotu64==1] = 0
intMotu64[intMotu64==2] = 1    
intMotu64[waters3==1] = 0

label_intMotu64, num_intMotu64, props_intMotu64  = identifyRegion(intMotu64)

#% creates a binary of the reef and removes interior reef pieces
intReef64 = np.copy(Lclean64)
intReef64[intReef64==2] = 1 
intReef64[waters3==1] = 0

label_intReef64, num_intReef64, props_intReef64  = identifyRegion(intReef64)

#% creates a filled atoll
filledAtoll = label(waters3)
filledAtoll[filledAtoll!=1] = 0
filledAtoll = 1 - filledAtoll

label_atoll_img, num_atoll, props_atoll = identifyRegion(filledAtoll)
atollCentroid = props_atoll[1].centroid
#save the new images
np.savetxt('waters3.csv',waters3,delimiter=',')
np.savetxt('intReef64.csv',intReef64,delimiter=',')
np.savetxt('intMotu64.csv',intMotu64,delimiter=',')

#%%Identify & save small objects
intMotu8 = np.copy(Lclean8)
intMotu8[intMotu8==1] = 0
intMotu8[intMotu8==2] = 1    
intMotu8[waters3==1] = 0
intMotuSmall = np.copy(intMotu8)
intMotuSmall[intMotu8==1] = intMotu8[intMotu8==1] - intMotu64[intMotu8==1]

label_intMotuSmall, num_intMotuSmall, props_intMotuSmall  = identifyRegion(intMotuSmall)

#% creates a binary of the reef and removes interior reef pieces
intReef64 = np.copy(Lclean64)
intReef64[intReef64==2] = 1 
intReef64[waters3==1] = 0

label_intReef64, num_intReef64, props_intReef64  = identifyRegion(intReef64)
#% regionprops into Pandas dataframe where small motu are each one row
plt.imshow(intMotuSmall)
dl1 = {}
resolution = 30
k = 0
for i in range(0,num_intMotuSmall):
    reefFlatLabel = label_intReef64[props_intMotuSmall[i].coords[0][0],props_intMotuSmall[i].coords[0][1]]
    SmallmotuLabel = label_intMotuSmall[props_intMotuSmall[i].coords[0][0],props_intMotuSmall[i].coords[0][1]]
    SmallareaM = props_intMotuSmall[i].area*resolution*resolution
    SmallperimM = props_intMotuSmall[i].perimeter*resolution
    SmallcentroidM = props_intMotuSmall[i].centroid
    if SmallareaM < 64*resolution*resolution and intMotuSmall[props_intMotuSmall[i].coords[0][0],props_intMotuSmall[i].coords[0][1]] == 1:
        m = {'small motu index': i, 'small water label': SmallmotuLabel,'reef flat label': reefFlatLabel, 'area (m^2)': SmallareaM, 'perimeter (m)': SmallperimM, 'centroid':SmallcentroidM }
        dl1[i] = m
        k = k + 1
       
df2_small = pd.DataFrame.from_dict(dl1,orient='index') 
df2_small.to_csv('df_motu_small.csv')

#%% save atoll-level properties
plt.imsave('filledAtollImage.png',filledAtoll)
p = props_atoll[1].perimeter*resolution #calc. perimeter
areaequiv = p**2/(4*np.pi) 

#atoll scale, each perimeter point saved as a row in d?
da2 = {}

da1 = pd.DataFrame(atollOceanPerim)
da1.columns=['atoll perimeter point y','atoll perimeter point x']
da1['o/l label'] = 'ocean'
da2 = pd.DataFrame(atollLagoonPerim)
da2.columns=['atoll perimeter point y','atoll perimeter point x']
da2['o/l label'] = 'lagoon'
datoll = da1.append(da2).reset_index()

datoll['atoll area (m2)'] = props_atoll[1].area*resolution*resolution
datoll['atoll perimeter outer (m)'] = props_atoll[1].perimeter*resolution
datoll['atoll major axis length (m)']=props_atoll[1].major_axis_length*resolution
datoll['atoll minor axis length (m)']= props_atoll[1].minor_axis_length*resolution
datoll['atoll diameter of a circle with the same area (m)']=props_atoll[1].equivalent_diameter*resolution
datoll['atoll area of circle with same perimeter (m2)']=areaequiv
c = props_atoll[1].centroid
cc = ((c, )*len(datoll))
datoll['centroid atoll'] = cc
datoll['centroid atoll x'] = c[1]
datoll['centroid atoll y'] = c[0]

datoll[['Horton Form Factor','Miller Circularity Ratio','Schummans Elongation','Ellipcity Index']] = calcStoddart(A1 = datoll['atoll area (m2)'][0],A2 = datoll['atoll area of circle with same perimeter (m2)'][0],d1=datoll['atoll diameter of a circle with the same area (m)'][0],L1=datoll['atoll major axis length (m)'][0],c = datoll['centroid atoll'][0],perim=atollOceanPerim,)
#%% plot large motu
plt.imshow(label_intMotu64)
plt.scatter(y = len(label_intMotu64[:,0])-largeLagoonCentroidInt[0], x = largeLagoonCentroidInt[1])
label_intMotu64_lagoon_centroid= label_intMotu64[largeLagoonCentroidInt[0],largeLagoonCentroidInt[1]]
label_intMotu64_corner= label_intMotu64[0,0]
#%% regionprops into Pandas dataframe where lagoon is one row   ----- from largest to smallest as identified by the User
dl1 = {}
k = 0
for i in range(0,len(centerPoints)):
    waterLabel = label_waters3[centerPoints[i][1],centerPoints[i][0]] 
    waterIndex = waterLabel - 1
    centroidL = props_waters3[waterIndex].centroid
    areaL = props_waters3[waterIndex].area*resolution*resolution
    perimL = props_waters3[waterIndex].perimeter*resolution
    
    m = {'lagoon index': i, 'water index': waterIndex, 'water label': waterLabel, 'area (m^2)': areaL, 'perimeter L (m)': perimL, 'centroid':centroidL }
    dl1[i] = m
    k = k + 1
       
dl = pd.DataFrame.from_dict(dl1,orient='index') 
dl.to_csv('df_lagoon.csv')

#%% regionprops into Pandas dataframe where each perimeter point is one row for the motu
d2 = {}
k = 0
for i in range(1,num_intMotu64):
    reefFlatLabel = label_intReef64[props_intMotu64[i].coords[0][0],props_intMotu64[i].coords[0][1]]
    motuLabel = label_intMotu64[props_intMotu64[i].coords[0][0],props_intMotu64[i].coords[0][1]]
    areaM = props_intMotu64[i].area*resolution*resolution
    perimM = props_intMotu64[i].perimeter*resolution
    centroidM = props_intMotu64[i].centroid
    perimeterPointsM = perim_object(label_intMotu64, i)
    if (intMotu64[props_intMotu64[i].coords[0][0],props_intMotu64[i].coords[0][1]]) == 1:
        for j in range(len(perimeterPointsM[0])):
            m = {'motu perimeter point x': perimeterPointsM[0][j][1],'motu perimeter point y': perimeterPointsM[0][j][0],'motu index': i, 'motu label': motuLabel, 'reef flat label': reefFlatLabel, 'area m^2': areaM, 'perimeter m': perimM, 'centroid':centroidM }
            d2[k] = m
            k = k + 1
        if len(perimeterPointsM) == 2 and (not label_intMotu64_lagoon_centroid == label_intMotu64_corner):
            for j in range(len(perimeterPointsM[1])):
                m = {'motu perimeter point x': perimeterPointsM[1][j][1],'motu perimeter point y': perimeterPointsM[1][j][0],'motu index': i, 'motu label': motuLabel, 'reef flat label': reefFlatLabel, 'area m^2': areaM, 'perimeter m': perimM, 'centroid':centroidM }
                d2[k] = m
                k = k + 1
df2 = pd.DataFrame.from_dict(d2,orient='index')    
#%% regionprops into Pandas dataframe where each perimeter point is one row for the reef flat
d3 = {}
k = 0
for i in range(1,num_intReef64):
    reefFlatLabel = label_intReef64[props_intReef64[i].coords[0][0],props_intReef64[i].coords[0][1]]
    areaR = props_intReef64[i].area*resolution*resolution
    perimR = props_intReef64[i].perimeter*resolution
    centroidR = props_intReef64[i].centroid
    perimeterPointsR = perim_object(label_intReef64, i)
    if (intReef64[props_intReef64[i].coords[0][0],props_intReef64[i].coords[0][1]]) == 1:
        for w in range(len(perimeterPointsR)):
            for j in range(len(perimeterPointsR[w])):
                m = {'reef flat perimeter point x': perimeterPointsR[w][j][1],'reef flat perimeter point y': perimeterPointsR[w][j][0],'reef flat index': i, 'reef flat label': reefFlatLabel, 'area m^2': areaR, 'perimeter R': perimR, 'centroid':centroidR }
                d3[k] = m
                k = k + 1
    
df3 = pd.DataFrame.from_dict(d3,orient='index')

#%% determine if a point is an ocean or a lagoon point
df2[['o/l label','closest lagoon angle','last point']] = df2.groupby(by = 'motu index').apply(OceanLagoonOuter, a = atollOceanPerim, l = atollLagoonPerim, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y').reset_index().set_index('level_1').drop('motu index',axis=1)
df3[['o/l label','closest lagoon angle','last point']] = df3.groupby(by = 'reef flat index').apply(OceanLagoonReefOuter, a = atollOceanPerim, l = atollLagoonPerim, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y').reset_index().set_index('level_1').drop('reef flat index',axis=1)
# drop the repeating first/last point
df2 = df2[df2['last point'] == 0].reset_index(drop = True)
# reset the index in df2 and df3
df2 = df2.groupby(by = 'motu index').apply(orderPoints).reset_index(drop = True)
df3 = df3.groupby(by = 'reef flat index').apply(orderPoints).reset_index(drop = True)
#%% create the angle normal to the shoreline that points towards the lagoon
if len(df2.groupby('motu index').mean()[['motu label']]) == 1: # have to run differently if there is only one reef flat
    df2['shore normal lagoon angle'] = df2.groupby(by = 'motu index').apply(shoreNormalAngle, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'closest lagoon angle').transpose() 
else:
    df2['shore normal lagoon angle'] = df2.groupby(by = 'motu index').apply(shoreNormalAngle, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'closest lagoon angle').reset_index().set_index('level_1').drop('motu index',axis=1) 
if len(df3.groupby('reef flat index').mean()[['reef flat label']]) == 1: # have to run differently if there is only one reef flat
    df3['shore normal lagoon angle'] = df3.groupby(by = 'reef flat index').apply(shoreNormalAngle, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', s3 = 'closest lagoon angle').transpose()
else:
    df3['shore normal lagoon angle'] = df3.groupby(by = 'reef flat index').apply(shoreNormalAngle, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', s3 = 'closest lagoon angle').reset_index().set_index('level_1').drop('reef flat index',axis=1)    

#%% calculate exposure angle for from the atoll object centroid
df2['binning angle ac'] = (90 - np.arctan2((atollCentroid[0]) - df2['motu perimeter point y'], df2['motu perimeter point x'] - (atollCentroid[1]))*180/np.pi) % 360
df3['binning angle ac'] = (90 - np.arctan2((atollCentroid[0]) - df3['reef flat perimeter point y'], df3['reef flat perimeter point x'] - (atollCentroid[1]))*180/np.pi) % 360
df2['binning angle lc'] = (90 - np.arctan2((largeLagoonCentroid[0]) - df2['motu perimeter point y'], df2['motu perimeter point x'] - (largeLagoonCentroid[1]))*180/np.pi) % 360
df3['binning angle lc'] = (90 - np.arctan2((largeLagoonCentroid[0]) - df3['reef flat perimeter point y'], df3['reef flat perimeter point x'] - (largeLagoonCentroid[1]))*180/np.pi) % 360
df2['exposure angle'] = (df2['shore normal lagoon angle'] + 180) % 360
df3['exposure angle'] = (df3['shore normal lagoon angle'] + 180) % 360
# reef flat exposure angle vs bin and shore normal lagoon
#%% Binning Group - North, South, East , West
df3['bins ac'] = pd.cut(df3['binning angle ac'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df2['bins ac'] = pd.cut(df2['binning angle ac'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df3['bins lc'] = pd.cut(df3['binning angle lc'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df2['bins lc'] = pd.cut(df2['binning angle lc'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
#3%%reef flat width in front of motu
#df2[['motu to reef flat distance','reef point x','reef point y']] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(WidthOuter,m2 = df3[df3['o/l label'] == 'ocean'], s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'reef flat perimeter point x', s4 = 'reef flat perimeter point y', s5 = 'exposure angle',s6 = 'reef flat label', res = resolution)
#%%reef flat width in behind motu
df2[['motu lagoon to reef flat lagoon','reef point l x','reef point l y']] = df2[df2['o/l label'] == 'lagoon'].groupby(by = 'motu index').apply(WidthOuter,m2 = df3[df3['o/l label'] == 'lagoon'], s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'reef flat perimeter point x', s4 = 'reef flat perimeter point y', s5 = 'shore normal lagoon angle',s6 = 'reef flat label', res = resolution)
#%% reef flat width
df3[['reef flat width','lagoon point x','lagoon point y']] = df3[df3['o/l label'] == 'ocean'].groupby(by = 'reef flat index').apply(WidthOuter,m2 = df3[df3['o/l label'] == 'lagoon'], s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', s3 = 'reef flat perimeter point x', s4 = 'reef flat perimeter point y', s5 = 'shore normal lagoon angle',s6 = 'reef flat label', res = resolution)
#%% reef flat width motu
df3[['reef flat width motu','lagoon point x m','lagoon point y m']] = df3[df3['o/l label'] == 'ocean'].groupby(by = 'reef flat index').apply(WidthOuterR,m2 = df3[df3['o/l label'] == 'lagoon'], mo = df2[df2['o/l label'] == 'ocean'], s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', s3 = 'reef flat perimeter point x', s4 = 'reef flat perimeter point y', s5 = 'shore normal lagoon angle',s6 = 'reef flat label',s7 = 'motu perimeter point x', s8 = 'motu perimeter point y', res = resolution, angTarget = 7, distTarget = 10)
#%% motu width
df2[['motu width','lagoon point x','lagoon point y']] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(WidthOuter,m2 = df2[df2['o/l label'] == 'lagoon'], s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'motu perimeter point x', s4 = 'motu perimeter point y', s5 = 'shore normal lagoon angle',s6 = 'motu label', res = resolution, distTarget = 4)


#%%reef flat width in front of motu
df2[['motu to reef flat distance','reef point x','reef point y','reef point ind']] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(WidthOuterEf,m2 = df3[df3['o/l label'] == 'ocean'], s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', s3 = 'reef flat perimeter point x', s4 = 'reef flat perimeter point y', s5 = 'exposure angle',s6 = 'reef flat label', res = resolution)

#%% Effective Reef Flat width

df3['effective reef flat width'] = df3['reef flat width']
df3['motu present'] = np.nan
df3.loc[(df3['o/l label'] == 'ocean','motu present')] = 0
df3['motu index'] = np.nan
#%% for the effective reef flat sets the widths that are in the motu to reef flat width to the correct values
df2[['reef point ind','motu index','motu to reef flat distance']].apply(EfW, axis = 1)
#%% Effective RW 2

df2mean = df2.groupby('motu index').mean()[['reef flat label','motu to reef flat distance']]

df2mean.apply(EfW2_outer, m2 = df3, axis = 1)


#%% Binned Lengths
#a = df2['bins ac'].unique()
df2binned = df2.groupby('bins ac').mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width']]

a = df2[df2['o/l label'] == 'ocean']['bins ac'].unique()
if len(a) == 1:
    df2['total binned motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'bins ac').apply(LengthOuterReef, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).reset_index().drop('bins ac',axis=1).transpose()
else:
    df2['total binned motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'bins ac').apply(LengthOuterReef, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).reset_index().set_index('level_1').drop('bins ac',axis=1)

#%% run the length code for the motu (centerline, oceanside, lagoonside)
if len(df2.groupby('motu index').mean()[['motu label']]) == 1: # have to run differently if there is only one reef flat
    df2['motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(LengthOuter, s3 = 'lagoon point x', s4 = 'lagoon point y', s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution, rou = 2).transpose()
    df2['ocean side motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(LengthOuterSimple, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).transpose()
    df2['lagoon side motu length'] = df2[df2['o/l label'] == 'lagoon'].groupby(by = 'motu index').apply(LengthOuterSimple, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).transpose()
else:
    df2['motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(LengthOuter, s3 = 'lagoon point x', s4 = 'lagoon point y', s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution, rou = 2).reset_index().set_index('level_1').drop('motu index',axis=1)
    df2['ocean side motu length'] = df2[df2['o/l label'] == 'ocean'].groupby(by = 'motu index').apply(LengthOuterSimple, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).reset_index().set_index('level_1').drop('motu index',axis=1)
    df2['lagoon side motu length'] = df2[df2['o/l label'] == 'lagoon'].groupby(by = 'motu index').apply(LengthOuterSimple, s1 = 'motu perimeter point x', s2 = 'motu perimeter point y', res = resolution).reset_index().set_index('level_1').drop('motu index',axis=1)
##% run the length code for the reef flat (centerline, oceanside, lagoonside)
#df3['reef flat length']  = df3[df3['o/l label'] == 'ocean'].groupby(by = 'reef flat index').apply(LengthOuter, s3 = 'lagoon point x', s4 = 'lagoon point y', s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', res = resolution, rou = 2).reset_index().set_index('level_1').drop('reef flat index',axis=1)
a = df3[df3['o/l label'] == 'ocean']['reef flat index'].unique()
if len(a) == 1: # have to run differently if there is only one reef flat with ocean side points
    df3['ocean side reef flat length'] = df3[df3['o/l label'] == 'ocean'].groupby(by = 'reef flat index').apply(LengthOuterReef, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', res = resolution).transpose()
else:
    df3['ocean side reef flat length'] = df3[df3['o/l label'] == 'ocean'].groupby(by = 'reef flat index').apply(LengthOuterReef, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', res = resolution).reset_index().set_index('level_1').drop('reef flat index',axis=1)
df3['total binned reef flat length'] = df3[df3['o/l label'] == 'ocean'].groupby(by = 'bins ac').apply(LengthOuterReef, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', res = resolution).reset_index().set_index('level_1').drop('bins ac',axis=1)
df3['binned indexed reef flat length'] = df3[df3['o/l label'] == 'ocean'].groupby(['bins ac','reef flat index']).apply(LengthOuterReef, s1 = 'reef flat perimeter point x', s2 = 'reef flat perimeter point y', res = resolution).reset_index().set_index('level_2').drop(['bins ac','reef flat index'],axis=1)

#%% add lat long of centroid
xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
width, height = ds.RasterXSize, ds.RasterYSize
xmax = xmin + width * xpixel
ymin = ymax + height * ypixel

ds = None

d = df3['centroid'].iloc[0].split()
centroid = [float(i.strip(',').strip('(').strip(')')) for i in d]

centroid_latLong =  xmin + centroid[1] * xpixel, ymax + centroid[0] * ypixel

centroid_lat = ymax + centroid[0] * ypixel
centroid_long = xmin + centroid[1] * xpixel

# drop some Unnamed columns
unwanted = df2.columns[df2.columns.str.startswith('Unnamed')]
df2.drop(unwanted, axis=1, inplace=True)
unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
df3.drop(unwanted, axis=1, inplace=True)

df2[['centroid_lat']] = centroid_lat
df2[['centroid_long']] = centroid_long
df2[['min_lat']] = ymin
df2[['min_long']] = xmin
df2[['max_lat']] = ymax
df2[['max_long']] = xmax
df2[['ypixel_lat']] = ypixel
df2[['xpixel_long']] = xpixel

df3[['centroid_lat']] = centroid_lat
df3[['centroid_long']] = centroid_long
df3[['min_lat']] = ymin
df3[['min_long']] = xmin
df3[['max_lat']] = ymax
df3[['max_long']] = xmax
df3[['ypixel_lat']] = ypixel
df3[['xpixel_long']] = xpixel

datoll[['centroid_lat']] = centroid_lat
datoll[['centroid_long']] = centroid_long
datoll[['min_lat']] = ymin
datoll[['min_long']] = xmin
datoll[['max_lat']] = ymax
datoll[['max_long']] = xmax
datoll[['ypixel_lat']] = ypixel
datoll[['xpixel_long']] = xpixel

#%% Save dataframes
df3.to_csv('df_reef_flat.csv')
df2.to_csv('df_motu.csv')
datoll.to_csv('df_atollOnly.csv')

#% 
df2summary = df2.groupby('motu index').mean()[['motu label','reef flat label','area m^2','perimeter m','motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','motu length','ocean side motu length','lagoon side motu length']]
df2binned = df2.groupby('bins ac').mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','total binned motu length']]
df3summary = df3.groupby('reef flat index').mean()[['reef flat label','area m^2','perimeter R','reef flat width','effective reef flat width','ocean side reef flat length']]
df3binned = df3.groupby('bins ac').mean()[['reef flat width','effective reef flat width','total binned reef flat length']]

#%
df3binned['percent length covered by motu'] = df2binned['total binned motu length']/df3binned['total binned reef flat length'] *100
df3binned.to_csv('dfbinned_reef_flat.csv')
df2binned.to_csv('dfbinned_motu.csv')
df3summary.to_csv('dfsummary_reef_flat.csv')
df2summary.to_csv('dfsummary_motu.csv')

print(i, atollName)
#%% Excel output

df2summary = df2.groupby('motu index').mean()[['motu label','reef flat label','area m^2','perimeter m','motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','motu length','ocean side motu length','lagoon side motu length']]
df2binned = df2.groupby('bins ac').mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','total binned motu length']]
df3summary = df3.groupby('reef flat index').mean()[['reef flat label','area m^2','perimeter R','reef flat width','effective reef flat width','ocean side reef flat length']]
df3binned = df3.groupby('bins ac').mean()[['reef flat width','effective reef flat width','total binned reef flat length']]
#%
df2summary2 = df2.groupby('motu index').mean()[['motu label','reef flat label']]
df2summary2[['area (m^2)','perimeter (m)','mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)','motu length (m)','ocean side motu length (m)','lagoon side motu length (m)']] = df2.groupby('motu index').mean()[['area m^2','perimeter m','motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','motu length','ocean side motu length','lagoon side motu length']]
df2summary2[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)']] = df2.groupby('motu index').std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width',]]
df3summary2 = df3.groupby('reef flat index').mean()[['reef flat label']]
df3summary2[['area (m^2)','perimeter (m)','mean reef flat width (m)','mean effective reef flat width (m)','ocean side reef flat length (m)']] = df3.groupby('reef flat index').mean()[['area m^2','perimeter R','reef flat width','effective reef flat width','ocean side reef flat length']]
df3summary2[['std reef flat width (m)','std effective reef flat width (m)']] = df3.groupby('reef flat index').std()[['reef flat width','effective reef flat width']]
#% totals
d = {'ocean basin': [atollName[0]],'country code': [atollName[2:4]],'atoll name': [atollName[5:]]}
df2totals = pd.DataFrame(d)
df2totals['Number Motu'] = len(df2['motu index'].unique())
df2totals[['total motu area (m^2)','total motu perimeter (m)','total motu length (m)','total ocean side motu length (m)','total lagoon side motu length (m)']] = df2summary2.sum()[['area (m^2)','perimeter (m)','motu length (m)','ocean side motu length (m)','lagoon side motu length (m)']]

d = {'ocean basin': [atollName[0]],'country code': [atollName[2:4]],'atoll name': [atollName[5:]]}
df3totals = pd.DataFrame(d)
df3totals['Number Reef Flats'] = len(df3['reef flat index'].unique())
df3totals[['total reef flat area (m^2)','total reef flat perimeter (m)','total ocean side reef flat length (m)']] = df3summary2.sum()[['area (m^2)','perimeter (m)','ocean side reef flat length (m)']]

#% binned
df2binned2 = df2.groupby('bins ac').mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','total binned motu length']]
df2binned2.columns = [['mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)','total binned motu length (m)']]
df2binned2[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)']] = df2.groupby('bins ac').std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width']]

df3binned2 = df3.groupby('bins ac').mean()[['reef flat width','effective reef flat width','total binned reef flat length']]
df3binned2.columns = [['mean reef flat width (m)','mean effective reef flat width (m)','total binned reef flat length (m)']]
#df3binned2['percent length covered by motu (%)'] = df2binned2['total binned motu length (m)'].squeeze().divide(df3binned2['total binned reef flat length (m)'].squeeze(),fill_value = 0)*100
#%
dirct = 'East'
df3binned2['percent length covered by motu (%)'] = 0
df3binned2.loc[dirct,'percent length covered by motu (%)'] = df2binned2['total binned motu length (m)'].squeeze()/df3binned2.loc[dirct]['total binned reef flat length (m)']*100
df3binned2[['std reef flat width (m)','std effective reef flat width (m)']] = df3.groupby('bins ac').std()[['reef flat width','effective reef flat width']]

df3binned2.to_csv('dfbinned_reef_flat.csv')
df2binned2.to_csv('dfbinned_motu.csv')
df3summary2.to_csv('dfsummary_reef_flat.csv')
df2summary2.to_csv('dfsummary_motu.csv')

datollsummary = datoll.head(1).drop(columns=['atoll perimeter point x','index','atoll perimeter point y','o/l label','centroid atoll x','centroid atoll y','atoll diameter of a circle with the same area (m)'])
unwanted = datollsummary.columns[datollsummary.columns.str.startswith('Unnamed')]
datollsummary.drop(unwanted, axis=1, inplace=True)

#%
unwanted = df2.columns[df2.columns.str.startswith('Unnamed')]
df2.drop(unwanted, axis=1, inplace=True)
df2columnsdrop =['last point', 'reef point l x', 'reef point l y', 'lagoon point x', 'lagoon point y', 'reef point x', 'reef point y',	'reef point ind','Horton Form Factor Atoll','Miller Circ. Ratio Atoll', 'Schummans Elongation Ratio Atoll','Ellipticity Index Atoll', 'Atoll Major Axis (km)','Atoll Minor Axis (km)','lagoon width cross','ocean width cross','lagoon reef width','ocean reef width','atoll area km^2','atoll perimeter km','motu excentricity']
df2.drop(df2columnsdrop, axis=1, inplace=True)


unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
df3.drop(unwanted, axis=1, inplace=True)
df3columnsdrop =['last point', 'reef flat width motu','lagoon point x', 'lagoon point y','lagoon point x m','lagoon point y m','MIN reef flat width motu','MIN lagoon point x m','MIN lagoon point y m','reef flat width motu cross','reef flat width motu c','lagoon point x m c','lagoon point y m c','atoll area km^2','atoll perimeter km',]
df3.drop(df3columnsdrop, axis=1, inplace=True)

#%
df2columns =['motu perimeter point x', 'motu perimeter point y', 'motu index', 'motu label', 'reef flat label', 'area (m^2)',	'perimeter (m)', 'centroid', 'o/l label', 'closest lagoon angle', 'shore normal lagoon angle', 'binning angle ac', 'binning angle lc', 'exposure angle', 'bins ac', 'bins lc', 'motu lagoon to reef flat lagoon (m)', 'motu width (m)', 'motu to reef flat distance (m)', 'motu length (m)', 'total binned motu length (m)','ocean side motu length (m)', 'lagoon side motu length (m)', 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']
df2.columns = [df2columns]
df3columns =['reef flat perimeter point x', 'reef flat perimeter point y', 'reef flat index', 'reef flat label', 'area (m^2)', 'perimeter (m)',	'centroid', 'o/l label', 'closest lagoon angle', 'shore normal lagoon angle', 'binning angle ac', 'binning angle lc', 'exposure angle', 'bins ac', 'bins lc', 'reef flat width (m)', 'effective reef flat width (m)', 'motu present', 'motu index', 'ocean side reef flat length (m)', 'total binned reef flat length (m)', 'binned indexed reef flat length (m)', 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']
df3.columns = [df3columns]

df2totals[[ 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']] = df2.mean()[[ 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']]
df3totals[[ 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']] = df3.mean()[[ 'centroid_lat', 'centroid_long', 'min_lat', 'min_long', 'max_lat', 'max_long', 'ypixel_lat', 'xpixel_long']]
#%
 # Create some Pandas dataframes from some data.
with pd.ExcelWriter('df_output.xlsx') as writer: 
    workbook=writer.book
    worksheet=workbook.add_worksheet('Summary tables')
    writer.sheets['Summary tables'] = worksheet
    worksheet.write_string(0, 0, 'Totals') 
    df2totals.to_excel(writer, sheet_name='Summary tables', startrow = 1, index = False)
    df3totals.to_excel(writer, sheet_name='Summary tables', startrow = 4, index = False)

    worksheet.write_string(7, 0, 'Motu Summary Table')
    df2summary2.to_excel(writer, sheet_name='Summary tables', startrow = 8)
    worksheet.write_string(len(df2summary2) + 10, 0, 'reef Flat Summary Table')
    df3summary2.to_excel(writer, sheet_name='Summary tables', startrow = len(df2summary2) + 11)
    worksheet.write_string(len(df2summary2) + len(df3summary2) + 13, 0, 'binned motu Table')
    df2binned2.to_excel(writer, sheet_name='Summary tables', startrow = len(df2summary2) + len(df3summary2) + 14)
    worksheet.write_string(len(df2summary2) + len(df3summary2) + 21, 0, 'binned reef flat Table')
    df3binned2.to_excel(writer, sheet_name='Summary tables', startrow = len(df2summary2) + len(df3summary2) + 22)
    
    worksheet.write_string(len(df2summary2) + len(df3summary2) + len(df3binned2) + len(df2binned2)+21, 0, 'Summary Atoll Whole Morphometrics')
    datollsummary.to_excel(writer, sheet_name='Summary tables', startrow = len(df2summary2) + len(df3summary2) + len(df3binned2) + len(df2binned2)+22)
    
    
    df3.to_excel(writer, sheet_name='reef flat data')
    df2.to_excel(writer, sheet_name='motu data')
    datoll.to_excel(writer,sheet_name='atoll perimeter pt data')
