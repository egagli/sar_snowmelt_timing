import pystac
import pystac_client
import stackstac
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import hvplot.xarray
from dask.distributed import Client
import rioxarray
import os
import matplotlib.pyplot as plt
import ulmo
from datetime import datetime
import xarray as xr
import rioxarray as rxr
import warnings

def get_s1_rtc_stac(bbox_gdf,start_time='2015-01-01',end_time=datetime.today().strftime('%Y-%m-%d'),orbit_direction='all',polarization='gamma0_vv',collection='mycollection.json'):
    # GDAL environment variables for better performance
    os.environ['AWS_REGION']='us-west-2'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR' 
    os.environ['AWS_NO_SIGN_REQUEST']='YES'
    
    # Load STAC ItemCollection
    stac_items = pystac.ItemCollection.from_file(collection)
    items = [item.to_dict(transform_hrefs=False) for item in stac_items]

    stack = stackstac.stack(items,dtype='float32')
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    time_slice = slice(start_time,end_time)
    
    scenes = stack.sel(band=polarization).sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice)
    
    if orbit_direction == 'all':
        scenes = scenes
    else:
        scenes = scenes.where(scenes.coords['sat:orbit_state']==orbit_direction,drop=True)
        
    return(scenes)

def get_median_ndvi(bbox_gdf,start_time='2020-07-30',end_time='2020-09-09'):
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    URL = "https://earth-search.aws.element84.com/v0"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-s2-l2a-cogs"],
    datetime=f"{start_time}/{end_time}").get_all_items()
    
    stack = stackstac.stack(items)
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    cloud_cover_threshold = 20
    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]

    nir, red, = lowcloud.sel(band="B08"), lowcloud.sel(band="B04")
    ndvi = (nir-red)/(nir+red)
    
    time_slice_ndvi = slice(start_time,end_time)
    scenes_ndvi = ndvi.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice_ndvi).median("time", keep_attrs=True)
    frames_ndvi_all = scenes_ndvi.compute()
    return(frames_ndvi_all)


def plot_timeseries_by_elevation_bin(ts_ds,dem_ds,bin_size=100,ax=None,normalize_bins=False):
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()
    
    dem_projected_ds = dem_ds.rio.reproject_match(ts_ds) # squeeze??
    dem_projected_ds = dem_projected_ds.where(ts_ds!=np.nan) # here mask DEM by ts_ds
    
    bin_centers=list(range(int(math.floor(dem_projected_ds.max()/100)*100)-bin_size//2,int(math.ceil(dem_projected_ds.min()/100)*100),-bin_size))
    backscatter_full = []

    for i,bin_center in enumerate(bin_centers):
        ts_bin_ds = ts_ds.where(np.abs(dem_projected_ds - bin_center) < bin_size//2)
        with warnings.catch_warnings(): #catches np.nanmean empty slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            backscatter_ts_for_bin = np.nanmean(ts_bin_ds.data.reshape(ts_bin_ds.shape[0],-1), axis=1) 
        backscatter_full.append(list(backscatter_ts_for_bin))
        
    backscatter_df = pd.DataFrame(backscatter_full,index=bin_centers,columns=ts_ds.time)
    
    if normalize_bins == True:
          backscatter_df = ((backscatter_df.T-backscatter_df.T.min())/(backscatter_df.T.max()-backscatter_df.T.min())).T
    colors = ax.pcolormesh(pd.to_datetime(ts_ds.time), bin_centers, backscatter_df,cmap='inferno',edgecolors=(1.0, 1.0, 1.0, 0.3)) #,vmin=0,vmax=0.5
    cbar = f.colorbar(colors,ax=ax)
    
    if normalize_bins == False:
        lab = 'Mean Backscatter [Watts]'
    else:
        lab = 'Normalized (Elevation-wise) Backscatter'
    
    cbar.ax.set_ylabel(lab, rotation=270, labelpad=15)

    ax.set_xlabel('Time')
    ax.set_ylabel('Elevation [m]')
    return(ax)


def plot_hyposometry(ts_ds,dem_ds,bin_size=100,ax=None):
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()    
    dem_projected_ds = dem_ds.rio.reproject_match(ts_ds) # squeeze??
    dem_projected_ds = dem_projected_ds.where(ts_ds!=np.nan) # here mask DEM by ts_ds
    bin_edges=list(range(int(math.floor(dem_projected_ds.max()/100)*100)+bin_size,int(math.ceil(dem_projected_ds.min()/100)*100)-bin_size,-bin_size))
    ax.hist(dem_projected_ds.squeeze().isel(time=0).to_numpy().ravel(),bins=bin_edges[::-1],orientation='horizontal',histtype='bar',ec='k')
    ax.set_ylim([np.array(bin_edges).min(),np.array(bin_edges).max()])
    ax.set_xlabel('# of Pixels')
    ax.set_ylabel('Elevation [m]')
    ax.set_title('Hyposometry Plot')
    return(ax)