# Tools for Exploring Snowmelt timing using Sentinel-1 C-band SAR

This repository contains notebooks and tools to identify snowmelt timing using timeseries analysis of backscatter of Sentinel-1 C-band SAR. This toolbox makes use of [Analysis Ready Sentinel-1 Backscatter Imagery](https://registry.opendata.aws/sentinel-1-rtc-indigo/) (Sentinel-1 RTC data) hosted on AWS managed by Indigo Ag, Inc. Special thanks to [Scott Henderson](https://github.com/scottyhq) for his [aws-rtc-stac](https://github.com/relativeorbit/aws-rtc-stac) repository which allows the creation of static STAC catalogs used in this repository. 

Please see the [examples folder](https://github.com/egagli/sar_snow_melt_timing/tree/main/examples) for notebooks that demonstrate example analysis that can be done using this toolbox. 


## Example: demonstrate_all_functions.ipynb
I would recommend starting with the [demonstrate_all_functions.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/demonstrate_all_functions.ipynb) notebook which quickly goes through usage of all of the toolbox's most powerful functions over an example area (Mt. Rainier) in quick succession with some example analysis. For example, here is a figure generated using the plot_bs_ndsi_swe_precip_with_context() function shown in that notebook. 

<img width="1358" alt="timeseries" src="https://user-images.githubusercontent.com/67975937/177890042-b788c2be-130a-463d-a24f-7bbd8d4bc4df.png">

On the left side of the figure is our area of interest with a basemap for context. The closest SNOTEL station is is shown. The colormap indicates the day of year estimate for snowmelt runoff onset. The right side of the figure shows the area of interest averaged Sentinel-1 Backscatter (seperated by relative orbit), Sentinel-2 NDSI, SNOTEL SWE, SNOTEL snow depth, and SNOTEL precipitation (pink for snow, blue for rain).

## Example: westernUS_comparison.ipynb
The [westernUS_comparison.ipynb](https://github.com/egagli/sar_snow_melt_timing/blob/main/examples/westernUS_comparison.ipynb) notebook is based off of my current research and focuses on comparing snowmelt timing in alpine and glacier environments in the Western US. Here is an example figure generated in this notebook.

![Mt  Rainier_bigfig (5)](https://user-images.githubusercontent.com/67975937/177889453-1f25bf2d-c430-43ba-940e-6dd5545b42b0.png)

**A)** & **B)** Sentinel-1 SAR backscatter in dB over Mt. Rainier, WA, before and after the 2020 snowmelt season, respectively. **C)** Pixel-wise day of year snowmelt runoff date predicted by SAR backscatter time series analysis over Mt. Rainier. **D)** Backscatter time series binned by elevation fromfor 2017-2021 over Mt Rainier. Seasonal patterns emerge, with the backscatter minima in each elevation band indicating when the snowpack is supersaturated (usually suggesting onset of snowmelt runoff onset). **E)** Temporal window selected from D) which isolates the 2020 snowmelt season. Green ticks indicate a Sentinel-1 SAR acquisition. Note the interesting jagged peaks occurring April to July, suggesting instances or combinations of melting, refreezing, and rain on snow events. 

## Fall 2021 AGU Poster
Finally, here is my Fall 2021 AGU Poster which summarizes my progress as of November 2021. Watch the narration of my poster [here](https://www.youtube.com/watch?v=S0ZCVqRnoJ8). Stay tuned for my next progress update! 
![agu_poster_48x36](https://user-images.githubusercontent.com/67975937/177890573-5ba07f43-cee6-4a11-ac1c-530738424946.png)
