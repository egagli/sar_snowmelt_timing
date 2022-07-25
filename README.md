# Tools for Exploring Snowmelt timing using Sentinel-1 C-band SAR

This repository contains notebooks and tools to identify snowmelt timing using timeseries analysis of backscatter of Sentinel-1 C-band SAR. This toolbox makes use of [Analysis Ready Sentinel-1 Backscatter Imagery](https://registry.opendata.aws/sentinel-1-rtc-indigo/) (Sentinel-1 RTC data) hosted on AWS managed by Indigo Ag, Inc. Special thanks to [Scott Henderson](https://github.com/scottyhq) for his [aws-rtc-stac](https://github.com/relativeorbit/aws-rtc-stac) repository which allows the creation of static STAC catalogs used in this repository. 


## Examples


Please see the examples folder for notebooks that demonstrate example analysis that can be done using this toolbox. 

I would recommend starting with the demonstrate_all_functions.ipynb notebook which quickly goes through usage of all of the toolbox's most powerful functions in quick succession with some example analysis. For example, here is a figure generated using the plot_bs_ndsi_swe_precip_with_context() function shown in that notebook. 

<img width="1358" alt="timeseries" src="https://user-images.githubusercontent.com/67975937/177890042-b788c2be-130a-463d-a24f-7bbd8d4bc4df.png">

On the left side of the figure is our area of interest with a basemap for context. The closest SNOTEL station is is shown. The colormap indicates the day of year estimate for snowmelt runoff onset. The right side of the figure shows the area of interest averaged Sentinel-1 Backscatter (seperated by relative orbit), Sentinel-2 NDSI, SNOTEL SWE, SNOTEL snow depth, and SNOTEL precipitation (pink for snow, blue for rain).


The westernUS_comparison.ipynb notebook is based off of my current research and focuses on comparing snowmelt timing in alpine and glacier environments in the Western US. 
![Mt  Rainier_bigfig (5)](https://user-images.githubusercontent.com/67975937/177889453-1f25bf2d-c430-43ba-940e-6dd5545b42b0.png)


![agu_poster_48x36](https://user-images.githubusercontent.com/67975937/177890573-5ba07f43-cee6-4a11-ac1c-530738424946.png)
