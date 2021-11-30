# AWS RTC STAC Generator

[![STAC Browser](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/browse.yml/badge.svg)](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/browse.yml)
[![STAC Updater](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/update.yml/badge.svg)](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/update.yml)
[![STAC Validator](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/validate.yml/badge.svg)](https://github.com/relativeorbit/aws-rtc-stac/actions/workflows/validate.yml)
[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=PangeoBinder&message=AWS+us-west-2&color=orange)](https://aws-uswest2-binder.pangeo.io/v2/gh/pangeo-data/pangeo-docker-images/HEAD?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Frelativeorbit%252Faws-rtc-stac%26urlpath%3Dlab%252Ftree%252Faws-rtc-stac%252F%26branch%3Dmain) 

This is a template repository with some GitHub Actions to facilitate on-demand generation of static STAC Catalogs for Analysis Ready Sentinel-1 Backscatter Imagery, on the [AWS Public Data Registry](Analysis Ready Sentinel-1 Backscatter Imagery). The goal is to create STAC for a specific MGRS square so that you can easily visualize it and open a notebook to do some custom analysis with Xarray.

Example: https://github.com/relativeorbit/aws-rtc-12SYJ

1. Click 'Use this template'
2. Edit CONFIG.yml with the [MGRS tile](https://github.com/relativeorbit/aws-rtc-stac/blob/main/SENTINEL1_RTC_CONUS_GRID.geojson) you want to analyze
4. Go to the 'Actions' tab of your new repository
5. Run 'STAC Updater' to generate STAC metadata

## STAC Browser via GitHub pages

Example: https://relativeorbit.github.io/aws-rtc-12SYJ

1. Go to Settings-->Pages--gh_pages to enable serving Github Pages
1. Run the STAC Browser GitHub Action
1. Go to the corresponding website for the repository

## Run Jupyter Notebooks

1. Edit the README.md badges to point to the correct repositories
2. Click on the binder badge to run notebooks on an emphemeral server
