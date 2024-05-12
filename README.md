# Tools for exploring snowmelt timing using Sentinel-1 C-band SAR
 
 [![DOI](https://zenodo.org/badge/433593658.svg)](https://zenodo.org/badge/latestdoi/433593658)

This repository contains notebooks and tools to identify snowmelt timing using timeseries analysis of backscatter of Sentinel-1 C-band SAR. The newer tools in this toolbox use the [Sentinel 1 Radiometrically Terrain Corrected (RTC)](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) product hosted on Microsoft Planetary Computer.

**Click [here](#Quickstart) to get to the quickstart example.** For more advanced use cases, please see the [examples folder](https://github.com/egagli/sar_snow_melt_timing/tree/main/examples) for notebooks that demonstrate example analysis that can be done using this toolbox. Check out the rendered notebooks using [nbviewer](https://nbviewer.org/github/egagli/sar_snow_melt_timing/tree/main/examples/). 

Originally, this toolbox made use of [Analysis Ready Sentinel-1 Backscatter Imagery](https://registry.opendata.aws/sentinel-1-rtc-indigo/) (Sentinel-1 RTC data) hosted on AWS managed by Indigo Ag, Inc. 

I'm currently in the process of cleaning up this repo, adding new starter code, and making these tools pip installable. 


## Geophysical Research Letters paper: Capturing the Onset of Mountain Snowmelt Runoff Using Satellite Synthetic Aperture Radar

Check out the paper that introduces this toolbox [here](https://doi.org/10.1029/2023GL105303).

![tweet1](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/47c81c77-4567-405a-80ec-10d63af8254d)
![tweet2](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/08185464-199e-4b87-8a65-810449638440)

Slight correction: In the plain language summary, the version of record contains the text: 

- "Finally, from 2016 to 2022, we documented a shift toward snowmelt happening earlier in the year, which means earlier spring flow in rivers.".
  
This should be:

- "Finally, from 2016 to 2022, we documented a shift toward snowmelt happening **later** in the year, which means **later** spring flow in rivers."

to be consistent with our findings and the rest of our text. My apologies for the oversight. -eric


## Example: Large scale processing of snowmelt runoff onset

![snowmelt_timing_interactive](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/8f1d70cb-bc7d-4419-9cee-2bd3a178b790)


Gif of interactive snowmelt runoff onset map of the western US hosted [here](https://egagli.github.io/view_sar_snowmelt_timing_map/). Code to process individual MGRS tiles in [process_mgrs.ipynb](https://github.com/egagli/sar_snowmelt_timing/blob/main/examples/process_mgrs.ipynb). Built the interactive map using [this](https://github.com/egagli/view_sar_snowmelt_timing_map) repository, based on [Scott Henderson's template](https://github.com/scottyhq/share-a-raster). 

## Quickstart
![quickstart](https://github.com/egagli/sar_snowmelt_timing/assets/67975937/58a68fc0-54bb-4fcc-ba8f-f4eda67a0ae1)

Check out the [intro_example.ipynb](https://github.com/egagli/sar_snowmelt_timing/blob/main/examples/intro_example.ipynb) notebook for a simple use case!



Thanks for stopping by! Feel free to email me for questions/collaborations at egagli@uw.edu.

All my best,

eric
