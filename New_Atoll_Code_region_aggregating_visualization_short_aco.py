#%%---------------------Import python libaries-----------------------
#import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.patches

#%%------------------input data
atollCompLoc = 'G:\Shared drives\Ortiz Atolls Database\CompositesWithCount\AllAtolls' #Location of all Atoll Composites (currently the ones made in 2019)
atollComp = os.listdir(atollCompLoc)
morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
countryName = 'AllAtollsNew'
newAtollList = []
for i in range(len(atollComp)): 
    fileName = atollComp[i]
    atollName = fileName[0:-20]
    full_path = morphOutput + '\\' + countryName + '\\' + atollName
    if  os.path.exists(full_path): 
        os.chdir(full_path)
        if os.path.isfile('df_motu.csv'): 
            newAtollList.append(atollName) 
PF = []
for i in range(len(newAtollList)):
    if newAtollList[i][0:4] == 'P_PF':
        PF.append(newAtollList[i])


        
#%% Create large dataFrames
i = 0

atollName = newAtollList[i]
fileName = atollName + '50c50mCountClip2.tif'
resolution = 30
morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
countryName = 'AllAtollsNew'
full_path = morphOutput + '\\' + countryName + '\\' + atollName # create county and atoll directory if they do not exist
os.chdir(full_path) # set working directory to the atoll directory 
  
# read in dataframes
df3 = pd.read_csv('df_reef_flat.csv')
df2 = pd.read_csv('df_motu.csv')
dfatoll = pd.read_csv('df_atollOnly.csv')
df2small = pd.read_csv('df_motu_small.csv')
 
df3['ocean basin'] = atollName[0]
df3['country code'] = atollName[2:4]
df3['atoll name'] = atollName[5:]

df2['ocean basin'] = atollName[0]
df2['country code'] = atollName[2:4]
df2['atoll name'] = atollName[5:]

dfatoll['ocean basin'] = atollName[0]
dfatoll['country code'] = atollName[2:4]
dfatoll['atoll name'] = atollName[5:]

df2small['ocean basin'] = atollName[0]
df2small['country code'] = atollName[2:4]
df2small['atoll name'] = atollName[5:]

unwanted = df2.columns[df2.columns.str.startswith('Unnamed')]
df2.drop(unwanted, axis=1, inplace=True)

unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
df3.drop(unwanted, axis=1, inplace=True)

unwanted = dfatoll.columns[dfatoll.columns.str.startswith('Unnamed')]
dfatoll.drop(unwanted, axis=1, inplace=True)

unwanted = df2small.columns[df2small.columns.str.startswith('Unnamed')]
df2small.drop(unwanted, axis=1, inplace=True)


df2all = df2.copy(deep=True)
df3all = df3.copy(deep=True)
dfatollall = dfatoll.copy(deep=True)
df2smallall = df2small.copy(deep=True)

for i in range(1,155):
    atollName = newAtollList[i]
    fileName = atollName + '50c50mCountClip2.tif'
    resolution = 30
    morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
    countryName = 'AllAtollsNew'
    full_path = morphOutput + '\\' + countryName + '\\' + atollName # create county and atoll directory if they do not exist
    os.chdir(full_path) # set working directory to the atoll directory 
      
    # read in dataframes
    df3 = pd.read_csv('df_reef_flat.csv')
    df2 = pd.read_csv('df_motu.csv')
    dfatoll = pd.read_csv('df_atollOnly.csv')
    df2small = pd.read_csv('df_motu_small.csv')
    
    df3['ocean basin'] = atollName[0]
    df3['country code'] = atollName[2:4]
    df3['atoll name'] = atollName[5:]
    
    df2['ocean basin'] = atollName[0]
    df2['country code'] = atollName[2:4]
    df2['atoll name'] = atollName[5:]

    dfatoll['ocean basin'] = atollName[0]
    dfatoll['country code'] = atollName[2:4]
    dfatoll['atoll name'] = atollName[5:]
    
    df2small['ocean basin'] = atollName[0]
    df2small['country code'] = atollName[2:4]
    df2small['atoll name'] = atollName[5:]
    
    unwanted = df2.columns[df2.columns.str.startswith('Unnamed')]
    df2.drop(unwanted, axis=1, inplace=True)
    
    unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
    df3.drop(unwanted, axis=1, inplace=True)
    
    unwanted = dfatoll.columns[dfatoll.columns.str.startswith('Unnamed')]
    dfatoll.drop(unwanted, axis=1, inplace=True)
    
    unwanted = df2small.columns[df2small.columns.str.startswith('Unnamed')]
    df2small.drop(unwanted, axis=1, inplace=True)
    
    frames2 = [df2all, df2]
    frames3 = [df3all, df3]
    frames4 = [dfatollall, dfatoll]
    framessmall =  [df2smallall, df2small]
    
    df2all = pd.concat(frames2)
    df3all = pd.concat(frames3)
    dfatollall = pd.concat(frames4)
    df2smallall = pd.concat(framessmall)
 #%% save large dataframes
morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
countryName = 'AllAtollsNew'
full_path = morphOutput + '\\' + countryName + '\\Regional_Analysis'
os.chdir(full_path) # set working directory to the atoll directory

df2all.to_csv('df_motu_allACO.csv')
df3all.to_csv('df_reef_flat_allACO.csv')
dfatollall.to_csv('df_atollOnly_all.csv')
df2smallall.to_csv('df_smallmotu_all.csv')

#%% Alternatively if large dataframes exist, just read them in large dataframes
morphOutput = 'G:\Shared drives\Ortiz Atolls Database\MorphometricOutput' # Location that the output will be saved to
countryName = 'AllAtollsNew'
full_path = morphOutput + '\\' + countryName + '\\Regional_Analysis'
os.chdir(full_path) # set working directory to the atoll directory

df3all = pd.read_csv('df_reef_flat_allACO.csv')
df2all = pd.read_csv('df_motu_allACO.csv')
dfatollall = pd.read_csv('df_atollOnly_all.csv')
df2smallall = pd.read_csv('df_smallmotu_all.csv')
df_binned2 = pd.read_csv('French Polynesia' + ' df_binned.csv')

df3all['bins latitude'] = pd.cut(df3all['centroid_lat'], bins = [-25, -13, -3, 4, 15], labels = ['-25 to -13', '-13 to -3', '-3 to 4', '4 to 15'], ordered = False)
df2all['bins latitude'] = pd.cut(df2all['centroid_lat'], bins = [-25, -13, -3, 4, 15], labels = ['-25 to -13', '-13 to -3', '-3 to 4', '4 to 15'], ordered = False)
dfatollall['bins latitude'] = pd.cut(dfatollall['centroid_lat'], bins = [-25, -13, -3, 4, 15], labels = ['-25 to -13', '-13 to -3', '-3 to 4', '4 to 15'], ordered = False)
#%% decide on grouping (regional or all or other)

df3all['bins abs latitude'] = pd.cut(df3all['centroid_lat'].abs(), bins = [-1, 4.7, 14, 30], labels = ['low', 'mid', 'high'], ordered = False)
df2all['bins abs latitude'] = pd.cut(df2all['centroid_lat'].abs(), bins = [-1, 4.7, 14, 30], labels = ['low', 'mid', 'high'], ordered = False)


atoll_centroids = df3all.groupby(['atoll name']).mean()[['centroid_lat','centroid_long']]

region_bin = df3all.groupby(['atoll name']).first()[['country code']]
t2 = region_bin.groupby('country code').size()
df3all_PF = df3all[df3all['country code'] == 'PF']
df2all_PF = df2all[df2all['country code'] == 'PF']

# depending on plotting interest/grouping
# region_name = 'French Polynesia'
# df_reef = df3all_PF
# df_motu = df2all_PF

region_name = 'All Atolls'
df_reef = df3all
df_motu = df2all

#%% # create summary tables
df_motu_summary = df_motu.groupby(['atoll name','motu index']).first()[['ocean basin','country code','bins abs latitude']]
df_motu_summary[['motu label','reef flat label','centroid_lat']] = df_motu.groupby(['atoll name','motu index']).mean()[['motu label','reef flat label','centroid_lat']]
df_motu_summary[['area (m^2)','perimeter (m)','mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)','mean ocean reef width (m)', 'mean lagoon reef width (m)','motu length (m)','ocean side motu length (m)','lagoon side motu length (m)']] = df_motu.groupby(['atoll name','motu index']).mean()[['area m^2','perimeter m','motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width','motu length','ocean side motu length','lagoon side motu length']]
df_motu_summary[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)','std ocean reef width (m)', 'std lagoon reef width (m)']] = df_motu.groupby(['atoll name','motu index']).std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width']]

df_reef_summary = df_reef.groupby(['atoll name','reef flat index']).mean()[['reef flat label','centroid_lat']]
df_reef_summary[['area (m^2)','perimeter (m)','mean reef flat width (m)','mean effective reef flat width (m)','mean reef flat width motu (std)','ocean side reef flat length (m)']] = df_reef.groupby(['atoll name','reef flat index']).mean()[['area m^2','perimeter R','reef flat width','effective reef flat width','reef flat width motu','ocean side reef flat length']]
df_reef_summary[['std reef flat width (m)','std effective reef flat width (m)','std reef flat width motu (m)']] = df_reef.groupby(['atoll name','reef flat index']).std()[['reef flat width','effective reef flat width','reef flat width motu']]
#% totals
def NumberObjects(m, s1):
    mt =m.copy()
    num = len(mt[s1].unique())      
    return num
df_totals = df_motu.groupby('atoll name').first()[['ocean basin','country code','bins abs latitude']]
df_totals[['atoll centroid_lat', 'atoll centroid_long']] = df_motu.groupby('atoll name').mean()[['centroid_lat', 'centroid_long']]
df_totals['Number Motu'] = df_motu.groupby('atoll name').apply(NumberObjects,s1 = 'motu index')
df_totals['Number Reef Flat'] = df_reef.groupby('atoll name').apply(NumberObjects,s1 = 'reef flat index')
#%
df_totals[['total motu area (m^2)','total motu perimeter (m)','total motu length (m)','total ocean side motu length (m)','total lagoon side motu length (m)']] = df_motu_summary.groupby('atoll name').sum()[['area (m^2)','perimeter (m)','motu length (m)','ocean side motu length (m)','lagoon side motu length (m)']]
df_totals[['mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)']] = df_motu.groupby('atoll name').mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width']]
df_totals[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)']] = df_motu.groupby('atoll name').std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width',]]

df_totals[['total reef flat area (m^2)','total reef flat perimeter (m)','total ocean side reef flat length (m)']] = df_reef_summary.groupby('atoll name').sum()[['area (m^2)','perimeter (m)','ocean side reef flat length (m)']]
df_totals[['mean reef flat width (m)','mean effective reef flat width (m)']] = df_reef.groupby('atoll name',).mean()[['reef flat width','effective reef flat width']]
df_totals[['std reef flat width (m)','std effective reef flat width (m)']] = df_reef.groupby('atoll name').std()[['reef flat width','effective reef flat width']]

df_totals['percent reef flat length covered by motu (%)'] = df_totals['total ocean side motu length (m)']/df_totals['total ocean side reef flat length (m)'] *100
df_totals['percent reef flat area covered by motu (%)'] = df_totals['total motu area (m^2)']/df_totals['total reef flat area (m^2)'] *100
df_totals['bins latitude'] = pd.cut(df_totals['atoll centroid_lat'], bins = [-25, -13, -3, 4, 15], labels = ['-25 to -13', '-13 to -3', '-3 to 4', '4 to 15'], ordered = False)

df_totals.to_csv(region_name + ' df_totals_ACO.csv')

#%% Create binned large dataFrames
df_binned = df_reef.groupby(['atoll name','bins ac']).mean()[['centroid_lat', 'centroid_long','reef flat width','effective reef flat width','reef flat width motu','total binned reef flat length']]
df_binned.columns = [['atoll centroid_lat', 'atoll centroid_long','mean reef flat width (m)','mean effective reef flat width (m)','mean reef flat width motu (m)','total binned reef flat length (m)']]
df_binned[['bins abs latitude']] = df_reef.groupby(['atoll name','bins ac']).first()[['bins abs latitude']]
df_binned[['std reef flat width (m)','std effective reef flat width (m)']] = df_reef.groupby(['atoll name','bins ac']).std()[['reef flat width','effective reef flat width']]
df_binned[['mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)','mean ocean reef width (m)', 'mean lagoon reef width (m)','total binned motu length (m)']] = df_motu.groupby(['atoll name','bins ac']).mean()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width','total binned motu length']]
df_binned[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)','std ocean reef width (m)', 'std lagoon reef width (m)']] = df_motu.groupby(['atoll name','bins ac']).std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width']]
df_binned['percent reef flat length covered by motu (%)'] = df_binned['total binned motu length (m)'].squeeze().divide(df_binned['total binned reef flat length (m)'].squeeze(),fill_value = 0)*100
df_binned = df_binned.reset_index(drop = False)

df_binned.to_csv(region_name + ' df_binnedACO.csv')

#%% merge small and large motu
df_motu_summary_large =  df_motu_summary.reset_index(drop = False)
df2all_small2 = df2smallall.reset_index(drop = False)
maxMotu = df_motu_summary_large[['atoll name','motu index']].groupby('atoll name').max()

df2all_small2['motu index'] = df2all_small2['small motu index'] + maxMotu.loc[df2all_small2['atoll name']].reset_index(drop = 'atoll name').squeeze()

frames = [df2all_small2, df_motu_summary_large]
df_motu_summary_all = pd.concat(frames)
#%% create total motu summary
df_totals_all = df_motu.groupby('atoll name').first()[['ocean basin','country code','bins abs latitude']]
df_totals_all['Number Motu'] = df_motu_summary_all.groupby('atoll name').apply(NumberObjects,s1 = 'motu index')
df_totals_all[['total motu area (km^2)']] = df_motu_summary_all.groupby('atoll name').sum()[['area (m^2)']]/1000000
df_totals_all[['total motu perimeter (km)']] = df_motu_summary_all.groupby('atoll name').sum()[['perimeter (m)']]/1000
df_totals_all['Number Motu small'] = df2all_small2.groupby('atoll name').apply(NumberObjects,s1 = 'motu index')
df_totals_all[['motu area small (km^2)']] = df2all_small2.groupby('atoll name').sum()[['area (m^2)']]/1000000
df_totals_all[['motu perimeter small (km)']] = df2all_small2.groupby('atoll name').sum()[['perimeter (m)']]/1000
df_totals_all['Number Motu large'] = df_motu_summary_large.groupby('atoll name').apply(NumberObjects,s1 = 'motu index')
df_totals_all[['motu area large (km^2)']] = df_motu_summary_large.groupby('atoll name').sum()[['area (m^2)']]/1000000
df_totals_all[['motu perimeter large (km)']] = df_motu_summary_large.groupby('atoll name').sum()[['perimeter (m)']]/1000

df_totals_all.to_csv('AllMotuSummarySmallLargeMotu.csv')

#%%Motu summary data
df_reef['exposure bin'] = pd.cut(df_reef['exposure angle'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df_motu['exposure bin'] = pd.cut(df_motu['exposure angle'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)

df_motu_summary = df_motu.groupby(['atoll name','motu index']).first()[['ocean basin','country code','bins abs latitude','motu excentricity']]
df_motu_summary[['motu label','reef flat label','centroid_lat']] = df_motu.groupby(['atoll name','motu index']).mean()[['motu label','reef flat label','centroid_lat']]
df_motu_summary[['area (m^2)','perimeter (m)','mean motu to reef flat distance (m)','mean motu lagoon to reef flat lagoon (m)','mean motu width (m)','mean ocean reef width (m)', 'mean lagoon reef width (m)','motu length (m)','ocean side motu length (m)','lagoon side motu length (m)']] = df_motu.groupby(['atoll name','motu index']).mean()[['area m^2','perimeter m','motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width','motu length','ocean side motu length','lagoon side motu length']]
df_motu_summary[['std motu to reef flat distance (m)','std motu lagoon to reef flat lagoon (m)','std motu width (m)','std ocean reef width (m)', 'std lagoon reef width (m)']] = df_motu.groupby(['atoll name','motu index']).std()[['motu to reef flat distance','motu lagoon to reef flat lagoon','motu width','ocean reef width', 'lagoon reef width']]
df_motu_summary[['directional bin']] = df_motu[df_motu['o/l label']=='ocean'].groupby(['atoll name','motu index'])['bins ac'].agg(pd.Series.mode).to_frame()
df_motu_summary[['exposure bin']] = df_motu[df_motu['o/l label']=='ocean'].groupby(['atoll name','motu index'])['exposure bin'].agg(pd.Series.mode).to_frame()
df_motu_summary['exposure bin'][df_motu_summary['exposure bin'].str.len() < 3.0] = np.nan
m = df_motu_summary[df_motu_summary['directional bin'] != df_motu_summary['exposure bin']]


#%% Exposure Angle & Position Angle
from scipy.stats import circmean
def circMean(m, s1):
    mt =m.copy()
    mt[[s1]]
    r = circmean(mt[[s1]], high = 360, low = 0)
    return r
    
df_motu_summary['mean exposure angle'] = df_motu[df_motu['o/l label']=='ocean'].groupby(['atoll name','motu index']).apply(circMean, s1 = 'exposure angle')
df_motu_summary['mean exposure bin'] = pd.cut(df_motu_summary['mean exposure angle'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df_motu_summary['mean position angle'] = df_motu[df_motu['o/l label']=='ocean'].groupby(['atoll name','motu index']).apply(circMean, s1 = 'binning angle ac')
df_motu_summary['mean position bin'] = pd.cut(df_motu_summary['mean position angle'], bins = [-1, 45, 135, 225, 315, 360], labels = ['North', 'East', 'South', 'West', 'North'], ordered = False)
df_merged = df_motu_summary.merge(df_reef_summary, on=['atoll name','reef flat label'])
#%% valuble column names
s1 = 'mean lagoon reef width (m)'
s2 = 'mean motu width (m)'
s3 = 'mean ocean reef width (m)'
s4 = 'motu total reef width (m)'
s5 = 'motu-reef-flat-dist / reef-flat width'
s6 = 'motu length / reef-flat length'

df_merged['motu total reef width (m)'] = df_merged[s1] + df_merged[s2] + df_merged[s3]
#x = motu length / atoll perimeter; y-axis = motu-reef-flat-dist / reef-flat width 
df_merged['motu-reef-flat-dist / reef-flat width'] = df_merged['mean ocean reef width (m)']/df_merged['motu total reef width (m)']
df_merged['motu length / reef-flat length'] = df_merged['motu length (m)']/df_merged['ocean side reef flat length (m)']
df_mergedm = df_merged[df_merged['mean position bin'] != df_merged['mean exposure bin']]
colors = {'low':'blue', 'mid':'orange', 'high':'green'}
p1 = s5
p2 = s6
#%% Plot critical reef flat width vs motu length FP
p1 = s3
p2 = 'motu length (m)'

cmp = plt.get_cmap('gist_earth',6)

ax1 = df_merged[(df_merged['mean position bin'] == 'North') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2, c= cmp(1), xlim = (0,70000), ylim = (0,3000), label = 'North',s=25)
df_merged[(df_merged['mean position bin'] == 'East') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2, c= cmp(2), xlim = (0,70000), ylim = (0,3000),ax=ax1, label = 'East',s=25)
df_merged[(df_merged['mean position bin'] == 'South') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2,  c= cmp(3), xlim = (0,70000), ylim = (0,3000),ax=ax1, label = 'South',s=10)
df_merged[(df_merged['mean position bin'] == 'West') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2,  c= cmp(4), xlim = (0,6000), ylim = (0,1500),ax=ax1, label = 'West',s=10)

plt.legend(framealpha=0.0)
plt.yticks(np.arange(0,1500,step=250),fontsize=12)
plt.xticks(np.arange(0,60000,step=15000),np.arange(0,60,step=15),fontsize=12)
plt.xlabel('Motu Length (km)')
plt.ylabel('Ocean Reef Width (m)')
ax1.tick_params(axis='both',which='major',width=2,length=7,direction='in')
#plt.savefig('MotuLengthOceanReefWidthFP.png',dpi=600)
#%% Plot critical reef flat width vs motu length normalized
#p1 = s3
#p2 = 'motu length (m)'
p1 = s5
p2 = s6

cmp = plt.get_cmap('gist_earth',6)

ax1 = df_merged[(df_merged['mean position bin'] == 'North') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2, c= cmp(1), xlim = (0,70000), ylim = (0,3000), label = 'North',s=25)
df_merged[(df_merged['mean position bin'] == 'East') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2, c= cmp(2), xlim = (0,70000), ylim = (0,3000),ax=ax1, label = 'East',s=25)
df_merged[(df_merged['mean position bin'] == 'South') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2,  c= cmp(3), xlim = (0,70000), ylim = (0,3000),ax=ax1, label = 'South',s=10)
df_merged[(df_merged['mean position bin'] == 'West') & (df_merged['country code']== 'PF')].plot.scatter(y=p1, x=p2,  c= cmp(4), xlim = (0,1), ylim = (0,1),ax=ax1, label = 'West',s=10)

plt.legend(framealpha=0.0)
plt.yticks(np.arange(0,1.1,step=.25))
plt.xticks(np.arange(0,1.1,step=.25))

plt.xlabel('Motu Length/Reef-flat Length')
plt.ylabel('Ocean Reef Width/Total Reef-flat Width')
ax1.tick_params(axis='both',which='major',width=2,length=7,direction='in')

#plt.savefig('MotuLengthOceanReefWidthFPNormalized.png',dpi=600)
#%%strings
df_merged = df_motu_summary.merge(df_reef_summary, on=['atoll name','reef flat label'])

s1 = 'mean lagoon reef width (m)'
s2 = 'mean motu width (m)'
s3 = 'mean ocean reef width (m)'
s4 = 'motu total reef width (m)'
s5 = 'motu-reef-flat-dist / reef-flat width'
s6 = 'motu length / reef-flat length'
df_merged['bins abs latitude'] = pd.cut(df_merged['centroid_lat_x'].abs(), bins = [-1, 4.7, 14, 30], labels = ['low', 'mid', 'high'], ordered = False)
df_merged['bins abs latitude'] = pd.cut(df_merged['centroid_lat_x'].abs(), bins = [-1, 4.7, 14, 30], labels = ['low', 'mid', 'high'], ordered = False)
#%%Motu length v reef width (m) binned by direction 

df_merged['motu total reef width (m)'] = df_merged[s1] + df_merged[s2] + df_merged[s3]
df_merged['motu-reef-flat-dist / reef-flat width'] = df_merged['mean ocean reef width (m)']/df_merged['motu total reef width (m)']
df_merged['motu length / reef-flat length'] = df_merged['motu length (m)']/df_merged['ocean side reef flat length (m)']

p1 = s3
p2 = 'motu length (m)'

blues = plt.get_cmap('Blues',5)
purples = plt.get_cmap('Purples',5)
reds = plt.get_cmap('Reds',5)
oranges = plt.get_cmap('Oranges',6)
greens = plt.get_cmap('Greens',5)

df_merged['bins abs lat'] = df_merged['bins abs latitude'].map({'high': 'high tropical', 'mid': 'mid tropical', 'low':'equatorial'})
ax1 = df_merged[df_merged['bins abs latitude'] == 'low'].plot.scatter(y=p1, x=p2, color=blues(3), label = 'equatorial')
df_merged[df_merged['bins abs latitude'] == 'mid'].plot.scatter(y=p1, x=p2, color=oranges(3), ax=ax1, label = 'mid tropical')
df_merged[df_merged['bins abs latitude'] == 'high'].plot.scatter(y=p1, x=p2,  color=greens(3), xlim = (0,70000), ylim = (0,3000),ax=ax1, label = 'high tropical')
plt.legend(framealpha=0.0)

plt.yticks(np.arange(0,3000,step=500))
plt.xticks(np.arange(0,70000,step=15000),np.arange(0,70,step=15))

# legend = plt.legend()
# legend.get_frame().set_facecolor('none')
plt.xlabel('Motu Length (km)')
plt.ylabel('Ocean Reef Width (m)')
ax1.tick_params(axis='both',which='major',width=2,length=7,direction='in')
#plt.savefig('MotuLengthOceanReefWidthAll.png',dpi=600)

#%% normalized All data critical reef width vs length

p1 = s5
p2 = s6

blues = plt.get_cmap('Blues',5)
purples = plt.get_cmap('Purples',5)
reds = plt.get_cmap('Reds',5)
oranges = plt.get_cmap('Oranges',6)
greens = plt.get_cmap('Greens',5)

df_merged['bins abs lat'] = df_merged['bins abs latitude'].map({'high': 'high tropical', 'mid': 'mid tropical', 'low':'equatorial'})

ax1 = df_merged[df_merged['bins abs latitude'] == 'low'].plot.scatter(y=p1, x=p2, color=blues(3), label = 'equatorial')
df_merged[df_merged['bins abs latitude'] == 'mid'].plot.scatter(y=p1, x=p2, color=oranges(3), ax=ax1, label = 'mid tropical')
df_merged[df_merged['bins abs latitude'] == 'high'].plot.scatter(y=p1, x=p2,  color=greens(3), xlim = (0,1), ylim = (0,1),ax=ax1, label = 'high tropical')
plt.legend(framealpha=0.0)

#plt.yticks(np.arange(0,1500,step=250),fontsize=12)
plt.yticks(np.arange(0,1.1,step=.25))
plt.xticks(np.arange(0,1.1,step=.25))

plt.xlabel('Motu Length/Reef-flat Length')
plt.ylabel('Ocean Reef Width/Total Reef-flat Width')
ax1.tick_params(axis='both',which='major',width=2,length=7,direction='in')
#plt.savefig('MotuLengthOceanReefWidthAllNorm.png',dpi=600)
#%% 2 d histigrams
# libraries


df_merged4 = df_merged.reset_index(drop = False)
df_merged4[['log 10 motu length (m)']] = np.log10(df_merged4[['motu length (m)']])
df_merged4[['log 10 motu width (m)']] = np.log10(df_merged4[['mean motu width (m)']])

df_merged5 = df_merged4[df_merged4['bins abs latitude'] == 'high'] #change to mid, low

#sns.displot(df_merged5, x='log 10 motu length (m)', y='log 10 motu width (m)', bins = [10,10])

#sns.displot(df_merged4, x='log 10 motu length (m)', y='log 10 motu width (m)', hue='bins abs latitude', kind="kde")
plt.xlim([0, 5])
plt.ylim([0, 5])

#sns.displot(df_merged4, x='motu length (m)', y='mean motu width (m)', hue='bins abs latitude', kind="kde")
sns.displot(df_merged5, x='log 10 motu length (m)', y='log 10 motu width (m)', hue='bins abs latitude',  kind="kde",fill = True, levels = (0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))

#%% all widths in one with the colors for FP
df_binned2['label bin'] = df_binned2['bins ac'].map({'North': 'a', 'East': 'b','South': 'c', 'West': 'd'})
df_binned2[['a) motu width','d) ocean reef width', 'c) lagoon reef width','b) reef flat width','e) effective reef flat width']] = df_binned2[['mean motu width (m)','mean ocean reef width (m)', 'mean lagoon reef width (m)','mean reef flat width (m)','mean effective reef flat width (m)']]

axs = df_binned2[['label bin','a) motu width','d) ocean reef width', 'c) lagoon reef width','b) reef flat width','e) effective reef flat width']].boxplot(by = 'label bin',figsize = (12,6),layout=(1, 5),patch_artist = True, grid=False, color = {'whiskers' : 'black',
                            'caps' : 'black',
                            'medians' : 'black',
                            'boxes' : 'black'})
cmp = plt.get_cmap('gist_earth',6)
for i in range(0,5):
    axs[i].findobj(matplotlib.patches.Patch)[0].set_facecolor(cmp(1))
    axs[i].findobj(matplotlib.patches.Patch)[1].set_facecolor(cmp(2))
    axs[i].findobj(matplotlib.patches.Patch)[2].set_facecolor(cmp(3))
    axs[i].findobj(matplotlib.patches.Patch)[3].set_facecolor(cmp(4))

axs[0].set_xticklabels(('North', 'East', 'South', 'West','North', 'East', 'South', 'West','North', 'East', 'South', 'West','North', 'East', 'South', 'West','North', 'East', 'South', 'West'))
axs[0].set(xlabel="", ylabel='mean width (m)')
axs[1].set(xlabel="")
axs[2].set(xlabel="")
axs[3].set(xlabel="")
axs[4].set(xlabel="")
plt.show()
#plt.savefig('WidthsFP_Boxplots.png')
#%%
df_merged['atoll name 2'] = df_merged.index
df_mergedbin = df_merged[['motu length (m)']]
df_mergedbin[['bins ac']] = df_merged[['mean position bin']]
df_mergedbin.reset_index(level=0, inplace=True)
df_binnedlength= df_mergedbin.groupby(['atoll name','bins ac']).mean()[['motu length (m)']]
df_binnedlength[['motu length (km)']] = df_binnedlength[['motu length (m)']]/1000
df_binnedlength.reset_index(level=1, inplace=True)
df_binnedlength['label bin'] = df_binnedlength['bins ac'].map({'North': 'a', 'East': 'b','South': 'c', 'West': 'd'})
#%% plot percent length blocked by motu binned box plot
df_binned2['label bin'] = df_binned2['bins ac'].map({'North': 'a', 'East': 'b','South': 'c', 'West': 'd'})

fig, ax = plt.subplots(1, 2, figsize=(8, 5))
df_binned2.boxplot('percent reef flat length covered by motu (%)','label bin', ax=ax[1],patch_artist = True, grid=False, color = {'whiskers' : 'black',
                            'caps' : 'black',
                            'medians' : 'black',
                            'boxes' : 'black'})
df_binnedlength.boxplot('motu length (km)','label bin', ax=ax[0],patch_artist = True, grid=False, color = {'whiskers' : 'black',
                            'caps' : 'black',
                            'medians' : 'black',
                            'boxes' : 'black'})

ax[0].set_xticklabels(('North', 'East', 'South', 'West'))
ax[1].set_xticklabels(('North', 'East', 'South', 'West'))
ax[1].set(xlabel="", ylabel='reef flat length blocked by motu (%)', title='percent reef flat blocked by motu')
ax[0].set(xlabel="", ylabel='mean motu length (km)', title='motu length')
cmp = plt.get_cmap('gist_earth',6)
for i in range(0,2):
    ax[i].findobj(matplotlib.patches.Patch)[0].set_facecolor(cmp(1))
    ax[i].findobj(matplotlib.patches.Patch)[1].set_facecolor(cmp(2))
    ax[i].findobj(matplotlib.patches.Patch)[2].set_facecolor(cmp(3))
    ax[i].findobj(matplotlib.patches.Patch)[3].set_facecolor(cmp(4))
ax[1].set_ylim((-1,120))
ax[0].set_ylim((-.4,44))
#plt.savefig('%BlockedFP_Boxplots.png')

#%%
df_motu_summary.to_csv(region_name + ' df_motu_summaryACO.csv')
df_reef_summary.to_csv(region_name + ' df_reef_summaryACO.csv')

#%% total perimeter/area by lattitude
df_reef['bins latitude 3'] = pd.cut(df_reef['centroid_lat'], bins = [-25,-23,-21,-19,-17,-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], labels = [-24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14], ordered = False)
df_motu['bins latitude 3'] = pd.cut(df_motu['centroid_lat'], bins = [-25,-23,-21,-19,-17,-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15], labels = [-24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14], ordered = False)

df_reef['bins latitude 4'] = pd.cut(df_reef['centroid_lat'], bins = [-25.5,-22.5,-19.5,-16.5,-13.5,-10.5,-7.5,-4.5,-1.5,1.5,4.5,7.5,10.5,13.5,16.5], labels = [-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15], ordered = False)
df_motu['bins latitude 4'] = pd.cut(df_motu['centroid_lat'], bins = [-25.5,-22.5,-19.5,-16.5,-13.5,-10.5,-7.5,-4.5,-1.5,1.5,4.5,7.5,10.5,13.5,16.5], labels = [-24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15], ordered = False)

s1 = 'bins latitude 4'

df_motu_summary[s1] = df_motu.groupby(['atoll name','motu index']).first()[[s1]]
df_reef_summary[s1] = df_reef.groupby(['atoll name','reef flat index']).first()[[s1]]
df_motu_summary = df_motu_summary.reset_index(drop=False)
df_lat_totals = df_motu_summary.groupby([s1]).sum()[['area (m^2)','perimeter (m)']]
df_lat_totals['number atolls'] = df_motu_summary.groupby([s1]).nunique()[['atoll name']]
df_lat_totals['number motu'] = df_motu_summary.groupby([s1]).count()[['area (m^2)']]
df_lat_totals['total motu area (km^2)'] = df_lat_totals['area (m^2)']/1000000
df_lat_totals['total motu perimeter (km)'] = df_lat_totals['perimeter (m)']/1000

df_lat_totals[['total reef flat area (m^2)','total reef flat perimeter (m)']] = df_reef_summary.groupby([s1]).sum()[['area (m^2)','perimeter (m)']]
df_lat_totals['number reef flat'] = df_reef_summary.groupby([s1]).count()[['area (m^2)']]
df_lat_totals['total reef flat area (km^2)'] = df_lat_totals['total reef flat area (m^2)']/1000000
df_lat_totals['total reef flat perimeter (km)'] = df_lat_totals['total reef flat perimeter (m)']/1000
df_lat_totals = df_lat_totals.drop(['area (m^2)','perimeter (m)','total reef flat area (m^2)','total reef flat perimeter (m)'], axis=1)
#%%
df_lat_totals2 = df_lat_totals.reset_index(drop = False)
df_lat_totals2 = df_lat_totals2.append({'bins latitude 4':-27,'number motu':0, 'total motu area (km^2)':0, 'total motu perimeter (km)':0, 'number reef flat':0, 'total reef flat area (km^2)':0,'total reef flat perimeter (km)':0},ignore_index=True)
df_lat_totals2 = df_lat_totals2.append({'bins latitude 4':15,'number motu':0, 'total motu area (km^2)':0, 'total motu perimeter (km)':0, 'number reef flat':0, 'total reef flat area (km^2)':0,'total reef flat perimeter (km)':0},ignore_index=True)
df_lat_totals2 = df_lat_totals2.sort_values([s1])
df_lat_totals2 = df_lat_totals2.reset_index(drop=True)
#%%
df2=df2all_PF
df3=df3all_PF

blues = plt.get_cmap('Blues',6)
purples = plt.get_cmap('Purples',6)
reds = plt.get_cmap('Reds',6)
oranges = plt.get_cmap('Oranges',6)
greens = plt.get_cmap('Greens',6)
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
lineW = 2
# Draw the density plot
sns.distplot(df2['motu width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'motu width', color = reds(4))

sns.distplot(df3['reef flat width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'reef flat width', color = blues(4))

sns.distplot(df2['lagoon reef width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'lagoon reef width', color = oranges(4))

sns.distplot(df2['ocean reef width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'ocean reef width', color = purples(4))

sns.distplot(df3['effective reef flat width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'effective reef flat width', color = greens(4))
    
# Plot formatting
plt.legend(prop={'size': 12}, title = 'Widths')
plt.title('a) French Polynesia Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.tight_layout()
#plt.savefig('DensityFP_AllWidths.png',dpi=600)

#%% density functions for the width measurements - motu width
df = df2.copy()
s2 = 'bins ac'

s1 = 'motu width'

#Draw the density plot
linecolor = reds
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.distplot(df[df[s2] == 'North'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             color = linecolor(5),
             label = 'North')

sns.distplot(df[df[s2] == 'East'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(4),
              label = 'East')

sns.distplot(df[df[s2] == 'South'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(3),
              label = 'South')

sns.distplot(df[df[s2] == 'West'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(2),
              label = 'West')
    
# Plot formatting

plt.title('b) Motu Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

#plt.savefig('DensityFP_motuwidth.png',dpi=600)

#%% density functions for the width measurements - reef flat width
df = df3.copy()
s2 = 'bins ac'

s1 = 'reef flat width'

#Draw the density plot
linecolor = blues
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.distplot(df[df[s2] == 'North'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             color = linecolor(5),
             label = 'North')

sns.distplot(df[df[s2] == 'East'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(4),
              label = 'East')

sns.distplot(df[df[s2] == 'South'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(3),
              label = 'South')

sns.distplot(df[df[s2] == 'West'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(2),
              label = 'West')
    
# Plot formatting

plt.title('c) Reef Flat Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)


plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

#plt.savefig('DensityFP_rfwidth.png',dpi=600)

#%% density functions for the width measurements - lagoon reef width
df = df2.copy()
s2 = 'bins ac'

s1 = 'lagoon reef width'

#Draw the density plot
linecolor = oranges
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.distplot(df[df[s2] == 'North'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             color = linecolor(5),
             label = 'North')

sns.distplot(df[df[s2] == 'East'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(4),
              label = 'East')

sns.distplot(df[df[s2] == 'South'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(3),
              label = 'South')

sns.distplot(df[df[s2] == 'West'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(2),
              label = 'West')
    
# Plot formatting

plt.title('d) Motu Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)


plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

#plt.savefig('DensityFP_motulagoonwidth.png',dpi=600)

#%% density functions for the width measurements - ocean reef width
df = df2.copy()
s2 = 'bins ac'

s1 = 'ocean reef width'

#Draw the density plot
linecolor = purples
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.distplot(df[df[s2] == 'North'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             color = linecolor(5),
             label = 'North')

sns.distplot(df[df[s2] == 'East'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(4),
              label = 'East')

sns.distplot(df[df[s2] == 'South'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(3),
              label = 'South')

sns.distplot(df[df[s2] == 'West'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(2),
              label = 'West')
    
# Plot formatting

plt.title('e) Ocean Reef Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')
plt.xlim([0, 2000])


leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

#plt.savefig('DensityFP_motuoceanwidth.png',dpi=600)
#%% density functions for the width measurements - effective reef width
df = df3.copy()
s1 = 'reef flat width'
s2 = 'bins ac'

s1 = 'effective reef flat width'

#df = df_motu[df_motu['motu length'] > 1000].copy()
# df = df2.copy()
# s1 = 'ocean reef width'
# s1 = 'lagoon reef width'
# s1 = 'motu width'
#Draw the density plot
linecolor = greens
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.distplot(df[df[s2] == 'North'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             color = linecolor(5),
             label = 'North')

sns.distplot(df[df[s2] == 'East'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(4),
              label = 'East')

sns.distplot(df[df[s2] == 'South'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(3),
              label = 'South')

sns.distplot(df[df[s2] == 'West'][s1], hist = False, kde = True,
              kde_kws = {'linewidth': lineW},
              color = linecolor(2),
              label = 'West')
    
# Plot formatting
#plt.legend(prop={'size': 12}, title = s1)
plt.title('f) Effective Reef Flat Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

plt.xlim([0, 2000])
plt.ylim([0,.013])
plt.yticks(np.arange(0,.015,step=.003))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

#plt.savefig('DensityFP_effectiverw.png',dpi=600)

#%% density functions for the width measurements - all atolls -
df2=df2all.copy()
df3=df3all.copy()

blues = plt.get_cmap('Blues',6)
purples = plt.get_cmap('Purples',6)
reds = plt.get_cmap('Reds',6)
oranges = plt.get_cmap('Oranges',6)
greens = plt.get_cmap('Greens',6)
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
lineW = 2
# Draw the density plot
sns.distplot(df2['motu width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'motu width', color = reds(4))

sns.distplot(df3['reef flat width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'reef flat width', color = blues(4))

sns.distplot(df2['lagoon reef width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'lagoon reef width', color = oranges(4))

sns.distplot(df2['ocean reef width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'ocean reef width', color = purples(4))

sns.distplot(df3['effective reef flat width'], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'effective reef flat width', color = greens(4))
    
# Plot formatting
plt.legend(prop={'size': 12}, title = 'Widths')
plt.title('a) All Atolls Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
plt.tight_layout()
#plt.savefig('DensityAll_AllWidths.png',dpi=600)

#%% density functions for the width measurements - all atolls - motu width
df = df2.copy()
s1 = 'motu width'
s2 = 'bins abs latitude'

linecolor = reds
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# Draw the density plot
sns.distplot(df[df[s2] == 'low'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},bins=int(2000),
             label = 'equatorial',color = linecolor(5))

sns.distplot(df[df[s2] == 'mid'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'mid tropical',color = linecolor(4))

sns.distplot(df[df[s2] == 'high'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'high tropical',color = linecolor(3))


# Plot formatting
plt.legend(prop={'size': 12}, title = s1)
plt.title('b) Motu Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

#plt.savefig('DensityAll_MotuWidths.png',dpi=600)
#%% density functions for the width measurements - all atolls - reef total width
df = df3.copy()
s1 = 'reef flat width'
s2 = 'bins abs latitude'


linecolor = blues
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# Draw the density plot
sns.distplot(df[df[s2] == 'low'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},bins=int(2000),
             label = 'equatorial',color = linecolor(5))

sns.distplot(df[df[s2] == 'mid'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'mid tropical',color = linecolor(4))

sns.distplot(df[df[s2] == 'high'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'high tropical',color = linecolor(3))


# Plot formatting
plt.legend(prop={'size': 12}, title = s1)
plt.title('c) Reef Flat Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

#plt.savefig('DensityAll_AllReefTotalWidths.png',dpi=600)

#%% density functions for the width measurements - all atolls - lagoon reef width
df = df2.copy()
s1 = 'lagoon reef width'
s2 = 'bins abs latitude'

linecolor = oranges
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# Draw the density plot
sns.distplot(df[df[s2] == 'low'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},bins=int(2000),
             label = 'equatorial',color = linecolor(5))

sns.distplot(df[df[s2] == 'mid'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'mid tropical',color = linecolor(4))

sns.distplot(df[df[s2] == 'high'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'high tropical',color = linecolor(3))

# Plot formatting
plt.legend(prop={'size': 12}, title = s1)
plt.title('d) Lagoon Reef Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

#plt.savefig('DensityAll_LagoonReefWidths.png',dpi=600)

#%% density functions for the width measurements - all atolls - ocean reef width
df = df2.copy()
s1 = 'ocean reef width'
s2 = 'bins abs latitude'

linecolor = purples
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# Draw the density plot
sns.distplot(df[df[s2] == 'low'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},bins=int(2000),
             label = 'equatorial',color = linecolor(5))

sns.distplot(df[df[s2] == 'mid'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'mid tropical',color = linecolor(4))

sns.distplot(df[df[s2] == 'high'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'high tropical',color = linecolor(3))


# Plot formatting
plt.legend(prop={'size': 12}, title = s1)
plt.title('e) Ocean Reef Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

#plt.savefig('DensityAll_OceanReefWidths.png',dpi=600)

#%% density functions for the width measurements - all atolls - effective width
df = df3.copy()
s1 = 'effective reef flat width'
s2 = 'bins abs latitude'

linecolor = greens
fig_dims = (4.5, 4)
fig, ax = plt.subplots(figsize=fig_dims)
# Draw the density plot
sns.distplot(df[df[s2] == 'low'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},bins=int(2000),
             label = 'equatorial',color = linecolor(5))

sns.distplot(df[df[s2] == 'mid'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'mid tropical',color = linecolor(4))

sns.distplot(df[df[s2] == 'high'][s1], hist = False, kde = True,
             kde_kws = {'linewidth': lineW},
             label = 'high tropical',color = linecolor(3))


# Plot formatting
plt.legend(prop={'size': 12}, title = s1)
plt.title('f) Effective Reef Flat Width')
plt.xlabel('Width (m)')
plt.ylabel('Density')

plt.xlim([0, 2000])
plt.ylim([0,.008])
plt.yticks(np.arange(0,.008,step=.0025))
plt.xticks(np.arange(0,2000,step=500))

plt.tick_params(axis='both',which='major',width=2,length=7,direction='in')
plt.tight_layout()

leg = plt.legend()
leg.get_frame().set_linewidth(0.0)

#plt.savefig('DensityAll_EffectivereefWidths.png',dpi=600)

#%% calc. critical reef-flat widths for diff groups

def calcCritWidth(df,s1,s2,l,s3,border):
    '''takes a dataframe, the two strings to iterate over, and the length to calc above, plus the bin order
        returns a dataframe with rows for each bin then each column: mean, std, number/count, %count, total'''
    aa =  df[df[s1]>l][s2].agg(['mean','std','count'])
    aa['total count'] = df.count().max()
    aa['percent count'] = aa['count']/aa['total count'] * 100
    df2 = pd.DataFrame([aa],index=['all'])
    for i in df[s3].dropna().unique():
        aa2 = df[(df[s3]==i) & (df[s1]>l)][s2].agg(['mean','std','count'])
        aa2['total count'] = df[df[s3]==i].count().max() #find total motu in given bin
        aa2['percent count'] = aa2['count']/aa2['total count'] * 100
        aa2.name=i
        df2 = df2.append([aa2])
    df2['length'] = l    
    df2 = df2.reindex(border)
    return df2

dfnewlong = calcCritWidth(df_merged,'motu length (m)','mean ocean reef width (m)',10000,'bins abs latitude',['low','mid','high','all'])

dfnew = calcCritWidth(df_merged,'motu length (m)','mean ocean reef width (m)',1000,'bins abs latitude',['low','mid','high','all'])

dfNew = dfnew.append(dfnewlong)

#now calc. for normalized values
dfnew = calcCritWidth(df_merged,'motu length / reef-flat length','motu-reef-flat-dist / reef-flat width',.1,'bins abs latitude',['low','mid','high','all'])
dfnewl = calcCritWidth(df_merged,'motu length / reef-flat length','motu-reef-flat-dist / reef-flat width',.25,'bins abs latitude',['low','mid','high','all'])

dfNewNorm = dfnew.append(dfnewl)
#%%
#df_mergedFP = df_merged #if you've reset way back in the beginning
#%%
# dfpnew = calcCritWidth(df_mergedFP,'motu length (m)','mean ocean reef width (m)',0,'directional bin',['North','East','South','West','all'])

# dfpnewl = calcCritWidth(df_mergedFP,'motu length (m)','mean ocean reef width (m)',10000,'directional bin',['North','East','South','West','all'])

# dfNewfp = dfpnew.append(dfpnewl)

# #now calc. for normalized values
# dfnew = calcCritWidth(df_mergedFP,'motu length / reef-flat length','motu-reef-flat-dist / reef-flat width',.1,'directional bin',['North','East','South','West','all'])
# dfnewl = calcCritWidth(df_mergedFP,'motu length / reef-flat length','motu-reef-flat-dist / reef-flat width',.25,'directional bin',['North','East','South','West','all'])

# dfNewNormFP = dfnew.append(dfnewl)

# #export these tables to excel
#  # Create some Pandas dataframes from some data.
# with pd.ExcelWriter('SummaryCriticalReefFlatWidth.xlsx') as writer: 
#     workbook=writer.book
#     worksheet=workbook.add_worksheet('All Motu')
#     writer.sheets['All Motu'] = worksheet
#     worksheet.write_string(0, 0, 'Totals critical reef flat width (m)') 
    
#     dfNew.to_excel(writer, sheet_name='All Motu', startrow = 1)
#     worksheet.write_string(13,0,'Normalized')
#     dfNewNorm.to_excel(writer, sheet_name='All Motu', startrow = 14)

#     worksheet=workbook.add_worksheet('FP Motu')
#     writer.sheets['FP Motu'] = worksheet
#     worksheet.write_string(0, 0, 'Totals critical reef flat width (m)') 
#     dfNewfp.to_excel(writer, sheet_name='FP Motu', startrow = 1)
#     worksheet.write_string(13,0,'Normalized')
#     dfNewNormFP.to_excel(writer, sheet_name='FP Motu', startrow = 14)
