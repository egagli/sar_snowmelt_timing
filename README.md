# Tools for exploring snowmelt timing using Sentinel-1 C-band SAR
 
 [![DOI](https://zenodo.org/badge/433593658.svg)](https://zenodo.org/badge/latestdoi/433593658)

This repository contains notebooks and tools to identify snowmelt timing using timeseries analysis of backscatter of Sentinel-1 C-band SAR. The newer tools in this toolbox use the [Sentinel 1 Radiometrically Terrain Corrected (RTC)](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) product hosted on Microsoft Planetary Computer.

Please see the [examples folder](https://github.com/egagli/sar_snow_melt_timing/tree/main/examples) for notebooks that demonstrate example analysis that can be done using this toolbox. Check out the rendered notebooks using [nbviewer](https://nbviewer.org/github/egagli/sar_snow_melt_timing/tree/main/examples/). 

Originally, this toolbox made use of [Analysis Ready Sentinel-1 Backscatter Imagery](https://registry.opendata.aws/sentinel-1-rtc-indigo/) (Sentinel-1 RTC data) hosted on AWS managed by Indigo Ag, Inc. 

I'm currently in the process of cleaning up this repo, adding new starter code, and making these tools pip installable.

## Geophysical Research Letters paper: Capturing the Onset of Mountain Snowmelt Runoff Using Satellite Synthetic Aperture Radar

Check out the paper that introduces this toolbox [here](https://doi.org/10.1029/2023GL105303).

![tweet1](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/47c81c77-4567-405a-80ec-10d63af8254d)
![tweet2](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/08185464-199e-4b87-8a65-810449638440)


## Example: Large scale processing of snowmelt runoff onset

![snowmelt_timing_interactive](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/8f1d70cb-bc7d-4419-9cee-2bd3a178b790)

<img width="899" alt="largescale" src="https://user-images.githubusercontent.com/67975937/231564758-e6d8a526-f723-44c9-b788-58a50a879750.png">

Screenshot of interactive snowmelt runoff onset map of the western US hosted [here](https://egagli.github.io/view_sar_snowmelt_timing_map/). Code to process individual MGRS tiles in [process_mgrs.ipynb](https://github.com/egagli/sar_snowmelt_timing/blob/main/examples/process_mgrs.ipynb). Built the interactive map using [this](https://github.com/egagli/view_sar_snowmelt_timing_map) repository, based on [Scott Henderson's template](https://github.com/scottyhq/share-a-raster). 

## (OLD) Example: visualize_all_volcanoes.ipynb

A study of snowmelt runoff onset on stratovcolcanoes in the Cascade Range: [visualize_all_volcanoes.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/visualize_all_volcanoes.ipynb) [[nbviewer link](https://nbviewer.org/github/egagli/sar_snow_melt_timing/blob/main/examples/visualize_all_volcanoes.ipynb)].

<img width="1092" alt="volc_ts" src="https://user-images.githubusercontent.com/67975937/231563176-1ce6ab96-91a1-49d7-bfad-2ffa736b3308.png">

A) Yearly Snowmelt Runoff Onset Maps for 10 Cascade Stratovolcanoes. B) Median snowmelt runoff onset at each elevation bin per year.

<img width="475" alt="fig4" src="https://user-images.githubusercontent.com/67975937/231562828-d56068c9-82f0-46b8-ba2c-68ff93fd2239.png">

A) 2015-2022 median snowmelt runoff onset maps. B) Median snowmelt runoff onset at each elevation bin with +/- 1 standard deviation.






## (OLD) Example: demonstrate_all_functions.ipynb
I would recommend starting with the [demonstrate_all_functions.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/demonstrate_all_functions.ipynb) [[nbviewer link](https://nbviewer.org/github/egagli/sar_snow_melt_timing/blob/main/examples/demonstrate_all_functions.ipynb)] notebook which quickly goes through usage of all of the toolbox's most powerful functions over an example area (Mt. Rainier) in quick succession with some example analysis. For example, here is a figure generated using the plot_bs_ndsi_swe_precip_with_context() function shown in that notebook. 

![mtrainier_timeseries](https://user-images.githubusercontent.com/67975937/180886014-7c87f643-9fa6-42c5-8d01-74190ee4826d.png)

On the left side of the figure is our area of interest with a basemap for context. The closest SNOTEL station is shown. The colormap indicates the day of year estimate for snowmelt runoff onset. The right side of the figure shows the AOI-averaged Sentinel-1 backscatter (seperated by relative orbit), AOI-averaged Sentinel-2 NDSI, SNOTEL SWE, SNOTEL snow depth, and SNOTEL precipitation (pink for snow, blue for rain).

## (OLD) Example: westernUS_comparison.ipynb
The [westernUS_comparison.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/westernUS_comparison.ipynb) [[nbviewer link](https://nbviewer.org/github/egagli/sar_snow_melt_timing/blob/main/examples/westernUS_comparison.ipynb)] notebook is based off of my current research and focuses on comparing snowmelt timing in alpine and glacier environments in the Western US. Here is an example figure generated in this notebook.

![combined_snowmelt_Mt  Rainier](https://user-images.githubusercontent.com/67975937/180885653-dbb241cf-de57-47eb-95e1-28b13b45e76e.png)

**A)** & **B)** Sentinel-1 SAR backscatter in dB over Mt. Rainier, WA, before and after the 2020 snowmelt season, respectively. **C)** Pixel-wise day of year snowmelt runoff date predicted by SAR backscatter time series analysis over Mt. Rainier. **D)** Backscatter time series binned by elevation for 2017-2021 over Mt Rainier. Seasonal patterns emerge, with the backscatter minima in each elevation band indicating when the snowpack is supersaturated (usually suggesting onset of snowmelt runoff). **E)** Temporal window selected from D) which isolates the 2020 snowmelt season. Green ticks indicate a Sentinel-1 SAR acquisition. Note the interesting jagged peaks occurring April to July, suggesting instances or combinations of melting, refreezing, and rain on snow events. 

## (OLD) Example: grandmesa.ipynb
The [grandmesa.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/grandmesa.ipynb) [[nbviewer link](https://nbviewer.org/github/egagli/sar_snow_melt_timing/blob/main/examples/grandmesa.ipynb)] notebook applies this toolbox over Grand Mesa, CO for SnowEx Hackweek 2022. I was on the [snowmelt-timing](https://github.com/snowex-hackweek/snowmelt-timing) team, and I handled the Sentinel-1 analysis of snowmelt timing for the 2020 melt season on Grand Mesa. See our final presentation video [here](https://www.youtube.com/watch?v=32q-hQ5pK48). Here are some figures generated from that notebook.


<p align="center">
  <img src="https://user-images.githubusercontent.com/67975937/180887533-610a3072-c1b1-4a31-9e2b-5056127346fd.png"/>
</p>


This figure shows the day of year snowmelt runoff onset date predicted by the SAR backscatter time series with the SnowEx Grand Mesa snowpit locations plotted on top.

![combined_grandmesa_crop](https://user-images.githubusercontent.com/67975937/180887541-7e71d5b1-defb-43ae-8a49-abfc663d5891.png)

This figure takes the runoff onset date map and zooms to each of the SnowEx Grand Mesa snowpit locations. The open snowpit sites (left-hand column) are a lot more spatially uniform in DOY estimates as expected. C-band radar will lose coherence in vegetated areas, so the backscatter minima at tree covered pixels will not necessarily be reflecting snowmelt properties. Everything seems to sync up nicely. We see minima in Sentinel-1 backscatter right as SWE seems to hit an inflection point. NDSI stays plateaued up until this point and drops significantly soon after. This is consistent for what we would expect throughout the melt season.

## (OLD) Fall 2021 AGU Poster
Finally, here is my Fall 2021 AGU Poster which summarizes my progress as of November 2021. Watch the narration of my poster [here](https://www.youtube.com/watch?v=S0ZCVqRnoJ8). Stay tuned for my next progress update! 
![agu_poster_48x36](https://user-images.githubusercontent.com/67975937/177890573-5ba07f43-cee6-4a11-ac1c-530738424946.png)

Thanks for stopping by! Feel free to email me for questions/collaborations at egagli@uw.edu.

All my best,

eric
