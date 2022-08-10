# AtollGeoMorph
Python scripts for semi-automated morphometric analysis of atolls from Landsat satellite Imagery.

The python scripts included allow users to create a temporal Landsat composite, classify and segment the image into land, water, reef-flat. Calculate morphometrics on each object, and finally visualize the resultant dataset.

1a. Call <git clone https://github.com/ale37911/AtollGeoMorph.git> in terminal to download the code. Then call < cd AtollGeoMorph > and call <conda env create -f atollGeomorp_env.yaml > to create the anaconda environment. Then activate the new conda environment with < conda activate atollGeoMorph >. Alternatively follow step 1b.

1b. atollGeoMorph_env.yaml - download this environment file to your computer to create an anaconda environment that includes all needed python libraries for running these scripts. Navigate to the location of the downloaded yaml file and type in the following command to create the conda environment and activate it.
  < conda env create --file=atollGeoMorph_env.yaml >
  < conda activate atollGeoMorph >
  
2. LandsatCompositeShort_ACO.ipynb - contains the jupyter notebook script for creating a temporal composite for an example atoll over a given time period and save the composite output to google drive. Once downloaded, navigate to the downloaded script and in the activated atollGeoMorph conda environment type the following and use jupyter notebook to run the script. You may need to activate google earth engine. 
  < jupyter notebook >
  
3. New_Atoll_Code_Pandas_Clean_ACO.py - contains the script to classify and segment the atoll composite (created with the previous ipynb file) into 3 classes, calculate simple and complex morphometrics on each object, and finally save all data as csv files for later analysis. This script assumes a location of atoll temporal composites and a location for resultant data analysis to be saved. The file structure is one folder per atoll labeled by the ocean basin of the atoll + country code + atoll name. All pandas dataframes created are saved as csv files in the atoll folder. In addition, geotif images are saved of temporal composite and classified and labeled image for further inquiry. Lastly, several small text files are created to aid with Lagoon delineation by the user to minimize repeated user inputs. This code includes a lot of functions for calculating morphometrics of the atoll.

4. New_Atoll_Code_region_aggregating_visualization_short_aco.py - contains a script to visualize the data created by the previous .py script for all atolls analyzed. It provides methods for aggreggating or group the dataset by a particular region or other metric. It also creates several summary dataframes collating all the data per atoll created in the previous script. 
