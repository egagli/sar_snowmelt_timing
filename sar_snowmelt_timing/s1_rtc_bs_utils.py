"""Library of functions to read and analyze Sentinel-1 C-band SAR data (RTC product from from https://registry.opendata.aws/sentinel-1-rtc-indigo/).

Author: Eric Gagliano (egagli@uw.edu)
Updated: 07/2022
"""

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
#import ulmo
from datetime import datetime
import xarray as xr
import rioxarray as rxr
import warnings
import geopandas as gpd
import rasterio as rio
import shapely
import scipy
import contextily as ctx
import planetary_computer
from pathlib import Path
import s1_rtc_bs_utils


def get_s1_rtc_stac(bbox_gdf,start_time='2015-01-01',end_time=datetime.today().strftime('%Y-%m-%d'),orbit_direction='all',polarization='gamma0_vv',collection='mycollection.json'):
    '''
    Returns a Sentinel-1 SAR backscatter xarray dataset using STAC data from Indigo over the given time and bounding box.

            Parameters:
                    bbox_gdf (geopandas GeoDataframe): geodataframe bounding box
                    start_time (str): start time of returned data 'YYYY-MM-DD'
                    end_time (str): end time of returned data 'YYYY-MM-DD'
                    orbit_direction (str): orbit direction of S1--can be all, ascending, or decending
                    polarization (str): SAR polarization, use gamma0_vv
                    collection (str): points to json collection, will be different for each MGRS square

            Returns:
                    scenes (xarray dataset): xarray stack of all scenes in the specified spatio-temporal window
    '''
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
    return scenes

def get_s1_rtc_stac_pc(bbox_gdf,start_time='2014-01-01',end_time=datetime.today().strftime('%Y-%m-%d'),polarization='vv',resolution=20):
    '''
    Returns a Sentinel-1 SAR backscatter xarray dataset using STAC data from Planetary computer over the given time and bounding box.

            Parameters:
                    bbox_gdf (geopandas GeoDataframe): geodataframe bounding box
                    start_time (str): start time of returned data 'YYYY-MM-DD'
                    end_time (str): end time of returned data 'YYYY-MM-DD'
                    orbit_direction (str): orbit direction of S1--can be all, ascending, or decending
                    polarization (str): SAR polarization, use gamma0_vv
                    collection (str): points to json collection, will be different for each MGRS square

            Returns:
                    scenes (xarray dataset): xarray stack of all scenes in the specified spatio-temporal window
    '''
    
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
    bbox = bbox_gdf.total_bounds
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start_time}/{end_time}", limit=1000) #remove limit if needed
    items = search.item_collection()
    stack = stackstac.stack(items, bounds_latlon=bbox, epsg=32610, dtype='float32',chunksize=512, resolution=resolution) #put resolution back in when fixed
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    
    if polarization == 'all':
        scenes = stack.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
    else:
        scenes = stack.sel(band=polarization).sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
        
    scenes = scenes#.drop_duplicates(dim='time',keep='first')

    return scenes

def get_s1_rtc_stac_odc_pc(bbox_gdf,start_time='2019-01-01',end_time='2019-12-31',resolution=10,epsg="EPSG:32610", resampling=None):
    import odc.stac,odc

    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
    bbox = bbox_gdf.total_bounds
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start_time}/{end_time}",limit=1000)
    items = search.item_collection()

    ds = odc.stac.load(
    search.get_items(), 
    chunks={'x':512,'y':512}, 
    #bands={"vv","vh"},
    groupby = 'sat:absolute_orbit',
    crs=epsg, # do we need this?
    resampling=resampling,
    fail_on_error = False,
    resolution=odc.geo.Resolution(resolution, -resolution)).where(lambda x: x > 0, other=np.nan)
    
    bounding_box_utm_gf = bbox_gdf.to_crs(ds.rio.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    
    scenes = ds.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sortby('time')
    
    df = gpd.GeoDataFrame.from_features(items.to_dict())
    df = df.groupby(['sat:absolute_orbit']).first().sort_values('datetime')

    scenes = scenes.assign_coords({'sat:orbit_state':('time',df['sat:orbit_state'])})
    scenes = scenes.assign_coords({'sat:relative_orbit':('time',df['sat:relative_orbit'].astype('int16'))}).to_array(dim='band').transpose('time','band','y','x')
    
    scenes = scenes.assign_attrs({'resolution':resolution})
    
    return scenes.chunk((100,1,512,512)) # does this make things faster?

def get_s1_rtc_stac_odc_pc_vvvh(bbox_gdf,start_time='2019-01-01',end_time='2019-12-31',resolution=10,epsg="EPSG:32610"):
    import odc.stac,odc

    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
    bbox = bbox_gdf.total_bounds
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start_time}/{end_time}",limit=1000)
    items = search.item_collection()

    ds = odc.stac.load(
    search.get_items(), 
    chunks={'x':512,'y':512}, 
    bands=["vv","vh"],
    crs=epsg, # do we need this?
    resolution=odc.geo.Resolution(resolution, -resolution)).where(lambda x: x > 0, other=np.nan)
    
    bounding_box_utm_gf = bbox_gdf.to_crs(ds.rio.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    
    scenes = ds.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
    
    orbits = [scene.properties['sat:relative_orbit'] for scene in items.items]
    orbits_abs = [scene.properties['sat:absolute_orbit'] for scene in items.items]
    direction = [scene.properties['sat:orbit_state'] for scene in items.items]
    scenes = scenes.assign_coords({'sat:orbit_state':('time',direction)})
    scenes = scenes.assign_coords({'sat:relative_orbit':('time',orbits)})
    scenes = scenes.assign_coords({'sat:absolute_orbit':('time',orbits_abs)}).to_array(dim='band').transpose('time','band','y','x')
    
    scenes = scenes.assign_attrs({'resolution':resolution})
    #scenes = scenes.groupby('sat:absolute_orbit').first()
    
    return scenes.chunk((100,1,512,512)) # does this make things faster?

def get_s1_rtc_stac_odc_mgrs_pc(bbox_gdf,start_time='2019-01-01',end_time='2019-12-31',resolution=10):
    import odc.stac,odc
    import ast
    from odc.geo.geobox import GeoBox


    
    s = bbox_gdf.iloc[0] #geoseries

    GEOMETRY = s.geometry.__geo_interface__
    BOUNDS = ast.literal_eval(s.utm_bounds)
    EPSG = str(s.epsg)

    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
    
    bbox = bbox_gdf.total_bounds
    search = catalog.search(
    collections = ['sentinel-1-rtc'],
    intersects = GEOMETRY,
    query={"proj:epsg": {"eq": EPSG}}, # Prevent adjacent MGRS Squares
    sortby='+datetime', #ascending order +, descending order -
)
    
    items = search.get_all_items()

    gf = gpd.GeoDataFrame.from_features( items.to_dict(), crs='EPSG:4326')
    gf['stac_id'] = [item.id for item in items]
    gf['time'] = pd.to_datetime(gf.datetime) # Use standard xarray dimension name 'time' for index
    gf = gf.set_index('time').tz_localize(None).sort_index()

    GRID = GeoBox.from_bbox(BOUNDS, 
                 crs=EPSG,
                 resolution=resolution,       
                )    
    
    ds = odc.stac.load(
    items, 
    #groupby='sat:absolute_orbit',
    chunks={'x':512,'y':512},
    geobox=GRID)
    
    ds = ds.sortby('time')
    ds = ds.where(lambda x: x > 0, other=np.nan)
    
    #bounding_box_utm_gf = bbox_gdf.to_crs(ds.rio.crs)
    #xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]
    
    #scenes = ds.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
    
    #orbits = [scene.properties['sat:relative_orbit'] for scene in items.items]
    #direction = [scene.properties['sat:orbit_state'] for scene in items.items]
    
    ds = ds.assign_coords(relative_orbit=gf['sat:relative_orbit'],
                      orbit_state=gf['sat:orbit_state'],
                      stac_id=gf['stac_id'],
                     )
    
    ds = ds.rename({'relative_orbit': 'sat:relative_orbit','orbit_state': 'sat:orbit_state'})
    
    da = ds.to_array(dim='band').transpose('time','band','y','x')
    
    da = da.assign_attrs({'resolution':resolution})
    
    return da 



def plot_sentinel1_acquisitons(ts_ds,ax=None,start_date='2015-01-01',end_date=datetime.today().strftime('%Y-%m-%d'),textsize=8):
    
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()
    
    asc = ts_ds[ts_ds.coords['sat:orbit_state']=='ascending']
    desc = ts_ds[ts_ds.coords['sat:orbit_state']=='descending']

    #f,ax=plt.subplots(figsize=(30,7))
                
    ax.scatter(np.array(asc.time),asc['sat:relative_orbit'],label='Ascending',c='red')
    ax.scatter(np.array(desc.time),desc['sat:relative_orbit'],label='Descending',c='blue')

    for i, label in enumerate(list(pd.to_datetime(asc.time.values).strftime('%Y-%m-%d \n %H:%M:%S'))):
        plt.annotate(label, (asc.time.values[i], asc['sat:relative_orbit'][i]),fontsize=textsize,rotation=45)
    
    for i, label in enumerate(list(pd.to_datetime(desc.time.values).strftime('%Y-%m-%d \n%H:%M:%S'))):
        plt.annotate(label, (desc.time.values[i], desc['sat:relative_orbit'][i]),fontsize=textsize,rotation=45)
    
    ax.legend()
    
    if start_date != '2015-01-01':
        ax.set_xlim([start_date,end_date])
        
    ax.set_ylim([0,200])
    ax.set_title('Sentinel-1 Relative Orbits')
    ax.set_xlabel('Datetime [UTC]')
    plt.tight_layout()


def get_median_ndvi(ts_ds,start_time='2020-07-30',end_time='2020-09-09'):
    '''
    Returns the median ndvi of the area covered by a given xarray dataset using Sentinel 2 imagery given a specific temporal window. Good for building an ndvi mask.

            Parameters:
                    ts_ds (xarray dataset): the area we will return the median ndvi over
                    start_time (str): start time of returned data 'YYYY-MM-DD'
                    end_time (str): end time of returned data 'YYYY-MM-DD'

            Returns:
                    frames_ndvi_compute (xarray dataset): computed ndvi median of the Sentinel 2 stack, reprojected to the same grid as the input dataset
    '''
    # go from ds to lat lon here
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    box = shapely.geometry.box(*ds_4326.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[box])
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    #lower_lon, upper_lat, upper_lon, lower_lat = gdf.geometry.total_bounds

    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-2-l2a"],
    datetime=f"{start_time}/{end_time}").get_all_items()
    
    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])
    
    stack = stackstac.stack(items,epsg=epsg_code)
    
    if np.unique(stack['proj:epsg']).size>1:
        stack = stack[stack['proj:epsg']!=stack['epsg']]
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    cloud_cover_threshold = 20
    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]

    nir, red, = lowcloud.sel(band="nir"), lowcloud.sel(band="red")
    ndvi = (nir-red)/(nir+red)
    
    #if np.unique(ndvi['proj:epsg']).size>1:
    #    try:
    #        ndvi = ndvi[ndvi['proj:epsg']==ndvi['proj:epsg'][1]].compute()
    #    except: 
    #        ndvi = ndvi[ndvi['proj:epsg']==ndvi['proj:epsg'][0]].compute()
    #else:
    #    ndvi = ndvi.compute()
    
    time_slice_ndvi = slice(start_time,end_time)
    scenes_ndvi = ndvi.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice_ndvi).median("time", keep_attrs=True)
    scenes_ndvi = scenes_ndvi.rio.write_crs(stack.rio.crs)
    frames_ndvi_compute = scenes_ndvi.rio.reproject_match(ts_ds).compute()
    return frames_ndvi_compute

def get_median_ndwi(ts_ds,start_time='2020-01-01',end_time='2020-02-15'):
    '''
    Returns the median ndvi of the area covered by a given xarray dataset using Sentinel 2 imagery given a specific temporal window. Good for building an ndvi mask.

            Parameters:
                    ts_ds (xarray dataset): the area we will return the median ndvi over
                    start_time (str): start time of returned data 'YYYY-MM-DD'
                    end_time (str): end time of returned data 'YYYY-MM-DD'

            Returns:
                    frames_ndvi_compute (xarray dataset): computed ndvi median of the Sentinel 2 stack, reprojected to the same grid as the input dataset
    '''
    # go from ds to lat lon here
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    box = shapely.geometry.box(*ds_4326.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[box])
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    #lower_lon, upper_lat, upper_lon, lower_lat = gdf.geometry.total_bounds

    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-2-l2a"],
    datetime=f"{start_time}/{end_time}").get_all_items()
    
    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])
    
    stack = stackstac.stack(items,epsg=epsg_code)
    
    if np.unique(stack['proj:epsg']).size>1:
        stack = stack[stack['proj:epsg']!=stack['epsg']]
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    cloud_cover_threshold = 20
    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]

    nir, red, = lowcloud.sel(band="green"), lowcloud.sel(band="nir")
    ndvi = (nir-red)/(nir+red)
    
    #if np.unique(ndvi['proj:epsg']).size>1:
    #    try:
    #        ndvi = ndvi[ndvi['proj:epsg']==ndvi['proj:epsg'][1]].compute()
    #    except: 
    #        ndvi = ndvi[ndvi['proj:epsg']==ndvi['proj:epsg'][0]].compute()
    #else:
    #    ndvi = ndvi.compute()
    
    time_slice_ndvi = slice(start_time,end_time)
    scenes_ndvi = ndvi.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice_ndvi).median("time", keep_attrs=True)
    scenes_ndvi = scenes_ndvi.rio.write_crs(stack.rio.crs)
    frames_ndvi_compute = scenes_ndvi.rio.reproject_match(ts_ds).compute()
    return frames_ndvi_compute


def get_py3dep_dem(ts_ds):
    import py3dep
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    bbox = ds_4326.rio.bounds()
    dem = py3dep.get_map("DEM", bbox, resolution=10, geo_crs="epsg:4326", crs="epsg:3857")
    dem.name = "dem"
    dem.attrs["units"] = "meters"
    dem_reproject = dem.rio.reproject_match(ts_ds) 
    return dem_reproject

def get_py3dep_aspect(ts_ds):
    import py3dep
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    bbox = ds_4326.rio.bounds()
    dem = py3dep.get_map("Aspect Degrees", bbox, resolution=10, geo_crs="epsg:4326", crs="epsg:3857")
    dem.name = "aspect"
    dem.attrs["units"] = "degrees"
    dem_reproject = dem.rio.reproject_match(ts_ds)
    return dem_reproject

def get_py3dep_slope(ts_ds):
    import py3dep
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    bbox = ds_4326.rio.bounds()
    dem = py3dep.get_map("Slope Degrees", bbox, resolution=10, geo_crs="epsg:4326", crs="epsg:3857")
    dem.name = "slope"
    dem.attrs["units"] = "degrees"
    dem_reproject = dem.rio.reproject_match(ts_ds) 
    return dem_reproject

def get_dah(ts_ds):
    import py3dep
    # Diurnal Anisotropic Heating Index [Böhner and Antonić, 2009]
    # https://www.sciencedirect.com/science/article/abs/pii/S0166248108000081
    # DAH = cos(alpha_max-alpha)*arctan(beta) where alpha_max is slope aspect 
    # recieving maximum heating alpha is slope aspect, beta is slope aspect
    # in radians. adpated from: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017WR020799
    # https://avalanche.org/avalanche-encyclopedia/aspect/
    alpha_max = 202.5
    aspect = get_py3dep_aspect(ts_ds)
    slope = get_py3dep_slope(ts_ds)
    DAH = np.cos(np.deg2rad(alpha_max-aspect))*np.arctan(np.deg2rad(slope))
    DAH_reproject = DAH.rio.reproject_match(ts_ds)
    return DAH_reproject

# def get_worldcover(ts_ds):
    
# #     bbox = ts_ds.rio.transform_bounds(rio.crs.CRS.from_epsg(4326))
    
# #     catalog = pystac_client.Client.open(
# #     "https://planetarycomputer.microsoft.com/api/stac/v1",
# #     modifier=planetary_computer.sign_inplace,
# #     )

# #     search = catalog.search(
# #     collections=["esa-worldcover"],
# #     bbox=bbox,
# #     )

# #     items = list(search.get_items())
# #     #print(f"Returned {len(items)} Items")

# #     string = f'{ts_ds.rio.crs}'
# #     epsg_code = int(string[5:])
    
# #     stack_lc = stackstac.stack(items, bounds_latlon=bbox, epsg=epsg_code, resolution=ts_ds.resolution)#ts_ds.resolution or ts_ds.rio.resolution()[0]
    
# #     stack_lc = stack_lc.min(dim='time').squeeze()
    
# #     stack_lc = stack_lc.rio.write_crs(ts_ds.rio.crs)
# #     stack_lc = stack_lc.rio.reproject_match(ts_ds)
#     import odc.stac
#     bbox = ts_ds.rio.transform_bounds(rio.crs.CRS.from_epsg(4326))

#     catalog = pystac_client.Client.open(
#     "https://planetarycomputer.microsoft.com/api/stac/v1",
#     modifier=planetary_computer.sign_inplace,
#     )

#     search = catalog.search(
#     collections=["esa-worldcover"],
#     bbox=bbox,
#     )

#     items = list(search.get_items())
#     #print(f"Returned {len(items)} Items")

#     string = f'{ts_ds.rio.crs}'
#     epsg_code = int(string[5:])

#     print(epsg_code)
#     stack_lc = odc.stac.load(search.get_items(), epsg=epsg_code, bbox=bbox, resolution=ts_ds.resolution)#ts_ds.resolution or ts_ds.rio.resolution()[0]
#     #stack_lc = stack_lc['map'].min(dim='time').squeeze()
    
#     return stack_lc
#     #stack_lc = stack_lc.rename({'longitude': 'x','latitude': 'y'})
#     #stack_lc = stack_lc.rio.write_crs(ts_ds.rio.crs)
#     #stack_lc = stack_lc.rio.reproject_match(ts_ds)
    
#     #return stack_lc

def get_worldcover(ts_ds, return_classmap=False):
    # to get 2020 WC, change time=-1 to time=0
    import odc.stac
    
    bbox = ts_ds.rio.transform_bounds(rio.crs.CRS.from_epsg(4326))
    
    catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)

    search = catalog.search(
    collections=["esa-worldcover"],
    bbox=bbox,)
    
    items = list(search.get_items())
    class_list = items[0].assets["map"].extra_fields["classification:classes"]
    classmap = {
        c["value"]: {"description": c["description"], "hex": c["color-hint"]}
        for c in class_list
    }



    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])


    stack_lc = odc.stac.load(search.get_items(),crs=epsg_code,resolution=ts_ds.resolution,bbox=bbox,bands=["map"]).isel(time=-1)
    stack_lc = stack_lc['map'].rio.reproject_match(ts_ds, Resampling = rio.enums.Resampling.mode)
    
    if return_classmap == False:
        return stack_lc
    else:
        return stack_lc, classmap

def get_snowmask(ts_ds):
    

    #fn = Path(Path(os.path.abspath(s1_rtc_bs_utils.__file__)).parent.parent, 
    # 'input/SnowClass/westernUS_MODIS_snow_classes_byte.tif').as_posix()

    fn = Path(Path(os.path.abspath(s1_rtc_bs_utils.__file__)).parent.parent, 
     'input/SnowClass/global_MODIS_snow_classes_byte.tif').as_posix()
        
    snow_mask = rxr.open_rasterio(fn)
    bbox = ts_ds.rio.transform_bounds(rio.crs.CRS.from_epsg(4326))
    snow_mask_clip = snow_mask.sel(x=slice(bbox[0],bbox[2]),y=slice(bbox[3],bbox[1])).squeeze()  
    
    snow_mask_proj = snow_mask_clip.rio.reproject_match(ts_ds)
    
    return snow_mask_proj

def get_orbits_with_melt_season_coverage(ts_ds,num_acquisitions_during_melt_season=6):
    # NEED TO SEPERATE OUT VALID ORBITS FOR VV AND VH
    year = ts_ds.time[0].dt.year.values
    unique_full_coverage = []
    melt_season = slice(f'{year}-02-01',f'{year}-07-31')
    
    for orbit in np.unique(ts_ds['sat:relative_orbit']):
        if len(ts_ds[ts_ds['sat:relative_orbit'].compute()==orbit].sel(time=melt_season).time.values) > num_acquisitions_during_melt_season:
            # if obs dont have more than a 1 month gap
            unique_full_coverage.append(orbit)
    unique_full_coverage = np.array(unique_full_coverage)
    
    return unique_full_coverage

def remove_border_noise(ts_ds):
    #try: 
    #    ts_ds = ts_ds.where(ts_ds.sel(band='vv')>0.001)
    #except:
    ts_ds = ts_ds.where(ts_ds>0.006)
    return ts_ds

def get_runoff_onset(ts_ds,return_seperate_orbits_and_polarizations=False, num_acquisitions_during_melt_season=8):
    # NEED TO SEPERATE OUT VALID ORBITS FOR VV AND VH
    #for pol in list(ts_ds.band.values):
    orbits = get_orbits_with_melt_season_coverage(ts_ds,num_acquisitions_during_melt_season=num_acquisitions_during_melt_season)
        #print(orbits)
    ts_ds = ts_ds[ts_ds['sat:relative_orbit'].isin(orbits).compute()]
    
    ts_ds = remove_border_noise(ts_ds)
    
    runoffs_int64 = ts_ds.groupby('sat:relative_orbit').map(lambda c: c.idxmin(dim='time')).astype(np.int64)
    
    if return_seperate_orbits_and_polarizations==False: # if false (default), return median
        if 'band' in runoffs_int64.dims:
            runoffs_int64 = runoffs_int64.where(runoffs_int64>0).median(dim=['sat:relative_orbit','band'],skipna=True)
        else:
            runoffs_int64 = runoffs_int64.where(runoffs_int64>0).median(dim=['sat:relative_orbit'],skipna=True)
            
    return runoffs_int64.astype('datetime64[ns]')

def get_runoffs_onset(ts_ds):
    orbits = get_orbits_with_melt_season_coverage(ts_ds,num_acquisitions_during_melt_season=8)
    ts_ds = ts_ds[ts_ds['sat:relative_orbit'].isin(orbits)]
    runoffs = ts_ds.groupby('sat:relative_orbit').map(lambda c: c.idxmin(dim='time'))
    return runoffs

# def get_runoff_onset(ts_ds,return_seperate_orbits=False,return_seperate_polarizations=False,combine_orbits='median'):
#     ts_ds = ts_ds.fillna(9999)
#     mins_index_runoff = ts_ds.argmin(dim='time',skipna=False)
#     runoff_dates = ts_ds[mins_index_runoff.compute()].time #this is just to establish the shape of runoff_dates #add.compute()
    

#     unique_full_coverage = get_orbits_with_melt_season_coverage(ts_ds,num_acquisitions_during_melt_season=4)
    
#     runoff_dates = runoff_dates.expand_dims(dim={"orbit":unique_full_coverage},axis=2).copy()
    
#     print(f'Using relative orbits: {unique_full_coverage}')
#     for orbit in unique_full_coverage:
#         print(f'Calculating runoff onset map from relative orbit {orbit}...')
#         ts_ds_orbit = ts_ds[ts_ds['sat:relative_orbit']==orbit]
#         mins_info_runoff = ts_ds_orbit.argmin(dim='time',skipna=True)
#         runoff_ds = ts_ds_orbit[mins_info_runoff.compute()].time #.compute()
#         runoff_ds = runoff_ds.where(ts_ds_orbit.min(dim='time')!=9999)#commment this line out, testing
#         runoff_dates.loc[:,:,orbit] = runoff_ds#.where((ts_ds_orbit.sum(dim='time')>0) & (ts_ds_orbit.sum(dim='time')<9998))# commenting this out

        
#     if (return_seperate_orbits == False) & (return_seperate_polarizations == False):
#         reduced_dims = ['orbit','band']
#     elif (return_seperate_orbits == False) & (return_seperate_polarizations == True):
#         reduced_dims = 'orbit'
#     elif (return_seperate_orbits == True) & (return_seperate_polarizations == False):
#         reduced_dims = 'band'
        
    
#     if (return_seperate_orbits == False) | (return_seperate_polarizations == False):
#         if combine_orbits=='median':
#             runoff_dates = runoff_dates.astype(np.int64)
#             runoff_dates = runoff_dates.where(runoff_dates>0).median(dim=reduced_dims,skipna=True).astype('datetime64[ns]') # changed to nanmean
#         elif combine_orbits=='mean':
#             runoff_dates = runoff_dates.astype(np.int64)
#             runoff_dates = runoff_dates.where(runoff_dates>0).mean(dim=reduced_dims,skipna=True).astype('datetime64[ns]') # changed to nanmean
#     else:
#         runoff_dates = runoff_dates
        
#     #runoff_dates = runoff_dates.where(ts_ds.min(dim='time')!=9999)
#     #runoff_dates = runoff_dates.dropna(dim='orbit',how='all')
#     #runoff_dates = runoff_dates.where(ts_ds.dropna(dim='time',how='all'))
    
#     return runoff_dates

def get_ripening_onset(ts_ds,orbit='ascending'): # fix this
    ts_ds = ts_ds.fillna(9999)
    ts_ds = ts_ds.where(ts_ds.coords['sat:orbit_state']==orbit,drop=True)
    mins_info_ripening = ts_ds.differentiate(coord='time',datetime_unit='W').argmin(dim='time',skipna=False) # dt=week
    ripening_dates = ts_ds[mins_info_ripening].time
    ripening_dates = ripening_dates.where(ts_ds.isel(time=0)!=9999)
    return ripening_dates

def get_stats(ts_ds,multiple_years=False,dem=None,aspect=None,slope=None,dah=None):
    
    if multiple_years == True:
        start_year = ts_ds.time[0].dt.year.values
        end_year = ts_ds.time[-1].dt.year.values
        years = np.arange(start_year,end_year+1,1)
        time_slice = slice(f'{start_year}-01-01',f'{end_year}-12-31')
        ts_ds_clipped = ts_ds.sel(time=time_slice)
        runoff_dates_all = get_runoff_onset(ts_ds_clipped).dt.dayofyear
        for year in years:
            time_slice = slice(f'{year}-01-01',f'{year}-12-31')
            ts_ds_clipped = ts_ds.sel(time=time_slice)
            runoff_dates_all = runoff_dates_all + get_runoff_onset(ts_ds_clipped).dt.dayofyear
        runoff_dates = runoff_dates_all/len(years)
    else:
        runoff_dates = get_runoff_onset(ts_ds).dt.dayofyear
        
    
    if all(np.array(ts_ds.coords['sat:orbit_state']=='descending')):
        ripening_dates = get_ripening_onset(ts_ds,orbit='descending')
    else:
        ripening_dates = get_ripening_onset(ts_ds)
        
    if dem is None:
        dem_projected = get_py3dep_dem(ts_ds)
    else:
        dem_projected = dem
    if aspect is None:
        aspect_projected = get_py3dep_aspect(ts_ds)
    else:
        aspect_projected = aspect
    if slope is None:
        slope_projected = get_py3dep_slope(ts_ds)
    else:
        slope_projected = slope
    if dah is None:
        dah_projected = get_dah(ts_ds)
    else: 
        dah_projected = dah
        
    dates_df = pd.DataFrame(columns=['x','y','elevation','aspect','slope','dah','runoff_dates','ripening_dates'])
    a1, a2 = np.meshgrid(dem_projected.indexes['x'],dem_projected.indexes['y'])
    dates_df['x'] = a1.reshape(-1)
    dates_df['y'] = a2.reshape(-1)
    dates_df['elevation'] = dem_projected.data.reshape(-1)
    dates_df['aspect'] = aspect_projected.data.reshape(-1)
    dates_df['slope'] = slope_projected.data.reshape(-1)
    dates_df['dah'] = dah_projected.data.reshape(-1)
    dates_df['runoff_dates'] = runoff_dates.data.reshape(-1)
    dates_df['ripening_dates'] = ripening_dates.dt.dayofyear.data.reshape(-1)
    dates_df = dates_df.dropna()
    
    dates_mls_df = dates_df.filter(['elevation','dah','runoff_dates','ripening_dates'])
    
    predictors = np.append(np.ones_like([dates_df['runoff_dates']]).T,dates_mls_df.iloc[:,[0,1]].to_numpy(),axis=1)
    B,_,_,_ = scipy.linalg.lstsq(predictors, dates_mls_df.iloc[:,2])
    dates_df['runoff_prediction'] = predictors.dot(B)
    
    predictors = np.append(np.ones_like([dates_df['runoff_dates']]).T,dates_mls_df.iloc[:,[0,1]].to_numpy(),axis=1)
    B,_,_,_ = scipy.linalg.lstsq(predictors, dates_mls_df.iloc[:,3])
    dates_df['ripening_prediction'] = predictors.dot(B)
    
    dates_gdf = gpd.GeoDataFrame(dates_df,geometry=gpd.points_from_xy(dates_df['x'],dates_df['y'],crs=ts_ds.rio.crs))
    dates_gdf=dates_gdf.set_index(['y','x'])
    
    return dates_gdf



def plot_timeseries_by_elevation_bin(ts_ds,dem_ds,bin_size=100,ax=None,normalize_bins=False, add_line=False,add_colorbar=True):
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
            backscatter_ts_for_bin = np.nanmedian(ts_bin_ds.data.reshape(ts_bin_ds.shape[0],-1), axis=1) # can choose mean or median 
        backscatter_full.append(list(backscatter_ts_for_bin))
        
    backscatter_df = 10*np.log10(pd.DataFrame(backscatter_full,index=bin_centers,columns=ts_ds.time))
    
    
    
    if normalize_bins == True:
          backscatter_df = ((backscatter_df.T-backscatter_df.T.min())/(backscatter_df.T.max()-backscatter_df.T.min())).T
    colors = ax.pcolormesh(pd.to_datetime(ts_ds.time), bin_centers, backscatter_df,cmap='inferno') #,vmin=0,vmax=0.5 # ,edgecolors=(1.0, 1.0, 1.0, 0.1)
    
    if add_colorbar:
        cbar = f.colorbar(colors,ax=ax)
    
    if normalize_bins == False:
        lab = 'Mean Backscatter [dB]'
    else:
        lab = 'Normalized (Elevation-wise) Backscatter'
    
    if add_line == True:
        mins = pd.DataFrame(backscatter_full,index=bin_centers,columns=ts_ds.time).idxmin(axis=1)
        ax.plot(mins.values,mins.index,color='white',linestyle='--',alpha=0.5)
    
    if add_colorbar:
        cbar.ax.set_ylabel(lab, rotation=90, labelpad=15)

    ax.set_xlabel('Time')
    ax.set_ylabel('Elevation [m]')
    return ax

def plot_timeseries_by_dah_bin(ts_ds,dem_ds,bin_size=0.25,ax=None,normalize_bins=False):
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()
    
    dem_projected_ds = dem_ds.rio.reproject_match(ts_ds) # squeeze??
    dem_projected_ds = dem_projected_ds.where(ts_ds!=np.nan) # here mask DEM by ts_ds
    
    bin_centers=list(np.arange(-1+bin_size/2,1,bin_size))
    backscatter_full = []

    for i,bin_center in enumerate(bin_centers):
        ts_bin_ds = ts_ds.where(np.abs(dem_projected_ds - bin_center) < bin_size/2)
        with warnings.catch_warnings(): #catches np.nanmean empty slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            backscatter_ts_for_bin = np.nanmean(ts_bin_ds.data.reshape(ts_bin_ds.shape[0],-1), axis=1) 
        backscatter_full.append(list(backscatter_ts_for_bin))
        
    backscatter_df = pd.DataFrame(backscatter_full,index=bin_centers,columns=ts_ds.time)
    
    if normalize_bins == True:
          backscatter_df = ((backscatter_df.T-backscatter_df.T.min())/(backscatter_df.T.max()-backscatter_df.T.min())).T
    colors = ax.pcolormesh(bin_centers, pd.to_datetime(ts_ds.time), backscatter_df.T,cmap='inferno',edgecolors=(1.0, 1.0, 1.0, 0.3)) #,vmin=0,vmax=0.5
    cbar = f.colorbar(colors,ax=ax,location='top',orientation='horizontal')
    
    if normalize_bins == False:
        lab = 'Mean Backscatter [Watts]'
    else:
        lab = 'Normalized (DAH-wise) Backscatter'
    
    #cbar.ax.set_ylabel(lab, rotation=270, labelpad=15)
    
    ax.set_xlabel('Diurnal Anisotropic Heating Index')
    ax.set_ylabel('Time')
    return ax

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
    return ax


def plot_dah_bins(ts_ds,dem_ds,bin_size=0.25,ax=None):
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()    
    dem_projected_ds = dem_ds.rio.reproject_match(ts_ds) # squeeze??
    dem_projected_ds = dem_projected_ds.where(ts_ds!=np.nan) # here mask DEM by ts_ds
    bin_edges=list(np.arange(-1,1+bin_size,bin_size))
    ax.hist(dem_projected_ds.squeeze().isel(time=0).to_numpy().ravel(),bins=bin_edges[::1],orientation='vertical',histtype='bar',ec='k')
    ax.set_xlim([-1,1])
    ax.set_ylabel('# of Pixels')
    ax.set_xlabel('DAH')
    ax.set_title('DAH Index Histogram')
    return ax


def plot_backscatter_ts_and_ndvi(ts_ds,ndvi_ds):
    frames = ts_ds
    frames_ndvi_all = ndvi_ds
    mins_info = frames.argmin(dim='time',skipna=False)
    f,ax=plt.subplots(3,2,figsize=(20,10))
    frames[mins_info].time.dt.dayofyear.where(frames_ndvi_all.values<0.2).plot(ax=ax[0,0],cmap='twilight')
    ax[0,0].set_title('Runoff Date w/ No Vegetation \n (NDVI < 0.2)')
    frames[mins_info].time.dt.dayofyear.where(frames_ndvi_all.values>0.2).where(frames_ndvi_all.values<0.6).plot(ax=ax[1,0],cmap='twilight')
    ax[1,0].set_title('Runoff Date w/ Sparse to Moderate Vegetation \n (0.2 < NDVI < 0.6)')
    #frames[mins_info].time.dt.dayofyear.where(frames_ndvi_all.values>0.4).where(frames_ndvi_all.values<0.6).plot(ax=ax[2,0])
    #ax[2,0].set_title('Runoff Date w/ Moderate Vegetation \n (0.4 < NDVI < 0.6)')
    frames[mins_info].time.dt.dayofyear.where(frames_ndvi_all.values>0.6).plot(ax=ax[2,0],cmap='twilight')
    ax[2,0].set_title('Runoff Date w/ Dense Vegetation \n (NDVI > 0.6)')

    ax[0,0].set_aspect('equal')
    ax[1,0].set_aspect('equal')
    ax[2,0].set_aspect('equal')

    ax[0,1].plot(frames.where(frames_ndvi_all.values<0.2).time,frames.where(frames_ndvi_all.values<0.2).mean(dim=['x','y']))

    ax[1,1].plot(frames.where(frames_ndvi_all.values>0.2).where(frames_ndvi_all.values<0.6).time,frames.where(frames_ndvi_all.values>0.2).where(frames_ndvi_all.values<0.6).mean(dim=['x','y']))

    #ax[2,1].plot(frames.where(frames_ndvi_all.values>0.4).where(frames_ndvi_all.values<0.6).time,frames.where(frames_ndvi_all.values>0.4).where(frames_ndvi_all.values<0.6).mean(dim=['x','y']))

    ax[2,1].plot(frames.where(frames_ndvi_all.values>0.6).time,frames.where(frames_ndvi_all.values>0.6).mean(dim=['x','y']))

    ax[0,1].set_title('Backscatter Time Series')
    ax[1,1].set_title('Backscatter Time Series')
    ax[2,1].set_title('Backscatter Time Series')

    ax[0,1].set_ylabel('Backscatter [Watts]')
    ax[1,1].set_ylabel('Backscatter [Watts]')
    ax[2,1].set_ylabel('Backscatter [Watts]')

    ax[0,1].set_ylim([0,0.5])
    ax[1,1].set_ylim([0,0.5])
    ax[2,1].set_ylim([0,0.5])

    plt.tight_layout()
    
def find_closest_snotel(ts_ds):
    import ulmo
    sites_df=pd.DataFrame.from_dict(ulmo.cuahsi.wof.get_sites('https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'),orient='index').astype({'elevation_m': 'float'})
    locations = pd.json_normalize(sites_df['location']).astype({'latitude': 'float','longitude':'float'})
    sites_gdf = gpd.GeoDataFrame(sites_df[['code','name','elevation_m']], geometry=gpd.points_from_xy(locations.longitude, locations.latitude))
    
    sites_gdf = sites_gdf.set_crs('epsg:4326')
    sites_gdf = sites_gdf.to_crs(ts_ds.rio.crs)
    
    sites_gdf['distance_km'] = sites_gdf.distance(shapely.geometry.box(*ts_ds.rio.bounds()))/1000
    sites_gdf = sites_gdf.sort_values(by='distance_km')
    sites_gdf = sites_gdf[sites_gdf['distance_km'].notnull()]

    return sites_gdf

def plot_closest_snotel(ts_ds,distance_cutoff=30,ax=None):
    
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()    
    
    sites_gdf = find_closest_snotel(ts_ds) 
    
    ts_ds = 10*np.log10(ts_ds)
    ts_ds.isel(time=0).plot(ax=ax,cmap='gray',vmin=-20,vmax=-1,add_colorbar=False)
    sites_gdf = sites_gdf[sites_gdf['distance_km']<distance_cutoff]
    color = sites_gdf.plot(column='distance_km',ax=ax,vmax=distance_cutoff,legend=True,cmap='viridis_r',legend_kwds={'label':'Distance from Study Site [km]','orientation':'vertical','fraction':0.0466,'pad':0.02})
    minx, miny, maxx, maxy = ts_ds.rio.bounds()
    ax.set_xlim([minx-1000*distance_cutoff*1.2,maxx+1000*distance_cutoff*1.2])
    ax.set_ylim([miny-1000*distance_cutoff*1.2,maxy+1000*distance_cutoff*1.2])

    ctx.add_basemap(ax=ax, crs=sites_gdf.crs, source=ctx.providers.Stamen.Terrain)

    ax.set_title('SNOTEL Sites Around Study Site')
    plt.tight_layout(rect=[0, 0, 0.9, 0.90])

    for x, y, label1, label2, label3, label4 in zip(sites_gdf.geometry.x, sites_gdf.geometry.y, sites_gdf.name, sites_gdf.code, sites_gdf.distance_km, sites_gdf.elevation_m):
        ax.annotate(f'{label1} \n{label2} \nElevation:{label4:.0f} m \nProximity:{label3:.2f} km', xy=(x, y), xytext=(15, -30), textcoords="offset points", fontsize=10,bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'))
        
        
    #for x, y, label1, label2, label3 in zip(sites_gdf.geometry.x, sites_gdf.geometry.y, sites_gdf.name, sites_gdf.code, sites_gdf.distance_km):
        #ax.annotate(f'{label1} \n{label2} \n{label3:.2f} km', xy=(x, y), xytext=(15, -30), textcoords="offset points", fontsize=10,bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'))
    
    return ax

def get_snotel(site_code, variable_code='SNOTEL:SNWD_D', start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    #print(ulmo.cuahsi.wof.get_site_info(wsdlurl, sitecode)['series'].keys())

    #print(sitecode, variablecode, start_date, end_date)
    values_df = None
    try:
        #Request data from the server
        site_values = ulmo.cuahsi.wof.get_values(wsdlurl, site_code, variable_code, start=start_date, end=end_date)
        #Convert to a Pandas DataFrame   
        values_df = pd.DataFrame.from_dict(site_values['values'])
        #Parse the datetime values to Pandas Timestamp objects
        values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=True)
        #Set the DataFrame index to the Timestamps
        values_df = values_df.set_index('datetime')
        #Convert values to float and replace -9999 nodata values with NaN
        values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        #Remove any records flagged with lower quality
        values_df = values_df[values_df['quality_control_level_code'] == '1']
    except:
        print("Unable to fetch %s" % variable_code)

    return values_df

def get_closest_snotel_data(ts_ds,variable_code='SNOTEL:SNWD_D',distance_cutoff=30,closest=False,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    
    sites_df = find_closest_snotel(ts_ds)
    sites_df = sites_df[sites_df['distance_km']<distance_cutoff]
    
    values_dict = {}
    
    for site_code in sites_df['code']:
        try:
            new_site = get_snotel(f'SNOTEL:{site_code}', variable_code,start_date=start_date, end_date=end_date)
            values_dict[site_code] = new_site['value']
        except:
            print(f'{site_code} data missing')
        if closest == True:
            break
        
    site_data_df = pd.DataFrame.from_dict(values_dict)
    
    return site_data_df

def get_s2_ndsi(ts_ds,cloud_cover_threshold=20):
    '''
    Returns the ndsi time series of the area covered by a given xarray dataset using Sentinel 2 imagery

            Parameters:
                    ts_ds (xarray dataset): the area we will return the median ndsi over

            Returns:
                    scenes_ndsi_compute (xarray dataset): computed ndsi time series with same spatial grid and temporal bounds as as the input dataset
    '''
    # go from ds to lat lon here
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    box = shapely.geometry.box(*ds_4326.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[box])
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    #lower_lon, upper_lat, upper_lon, lower_lat = gdf.geometry.total_bounds

    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    start_time = pd.to_datetime(ts_ds.time[0].values).strftime('%Y-%m-%d')
    end_time = pd.to_datetime(ts_ds.time[-1].values).strftime('%Y-%m-%d')
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-2-l2a"],
    datetime=f"{start_time}/{end_time}").get_all_items()

    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])

    stack = stackstac.stack(items,bounds_latlon=(bbox_gdf.bounds.values[0]),epsg=epsg_code) #epsg=epsg_code
        
    if np.unique(stack['proj:epsg']).size>1:
        stack = stack[stack['proj:epsg']!=stack['epsg']]
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]
    lowcloud = lowcloud
    #lowcloud = lowcloud.drop_duplicates("time","first")
    # snow.groupby(snow.time.dt.date).mean() use this for groupby date
    vir, swir = lowcloud.sel(band="green"), lowcloud.sel(band="swir16")
    ndsi = (vir-swir)/(vir+swir)    
    
        
    time_slice = slice(start_time,end_time)
    scenes_ndsi = ndsi.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice)
    scenes_ndsi = scenes_ndsi.rio.write_crs(stack.rio.crs)
    
    
    scenes_ndsi_compute = scenes_ndsi.rio.reproject_match(ts_ds).resample(time='1D',skipna=True).mean("time", keep_attrs=True).dropna('time',how='all')#.compute() #what was this for again?????
    #scenes_ndsi_compute = scenes_ndsi_compute.where(ts_ds.isel(time=0)>0)
    return scenes_ndsi_compute

def get_s2_ndwi(ts_ds,cloud_cover_threshold=20,start_time=None,end_time=None):
    '''
    Returns the ndsi time series of the area covered by a given xarray dataset using Sentinel 2 imagery

            Parameters:
                    ts_ds (xarray dataset): the area we will return the median ndsi over

            Returns:
                    scenes_ndsi_compute (xarray dataset): computed ndsi time series with same spatial grid and temporal bounds as as the input dataset
    '''
    # go from ds to lat lon here
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    box = shapely.geometry.box(*ds_4326.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[box])
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    #lower_lon, upper_lat, upper_lon, lower_lat = gdf.geometry.total_bounds

    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    if start_time == None:
        start_time = pd.to_datetime(ts_ds.time[0].values).strftime('%Y-%m-%d')
    if end_time == None:
        end_time = pd.to_datetime(ts_ds.time[-1].values).strftime('%Y-%m-%d')
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-2-l2a"],
    datetime=f"{start_time}/{end_time}").get_all_items()

    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])

    stack = stackstac.stack(items,bounds_latlon=(bbox_gdf.bounds.values[0]),epsg=epsg_code) #epsg=epsg_code
        
    if np.unique(stack['proj:epsg']).size>1:
        stack = stack[stack['proj:epsg']!=stack['epsg']]
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]
    lowcloud = lowcloud
    #lowcloud = lowcloud.drop_duplicates("time","first")
    # snow.groupby(snow.time.dt.date).mean() use this for groupby date
    vir, swir = lowcloud.sel(band="nir"), lowcloud.sel(band="swir22")
    ndwi = (vir-swir)/(vir+swir)    
    
        
    time_slice = slice(start_time,end_time)
    scenes_ndwi = ndwi.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice)
    scenes_ndwi = scenes_ndwi.rio.write_crs(stack.rio.crs)
    
    
    scenes_ndwi_compute = scenes_ndwi.rio.reproject_match(ts_ds).resample(time='1D',skipna=True).mean("time", keep_attrs=True).dropna('time',how='all')#.compute() #what was this for again?????
    #scenes_ndsi_compute = scenes_ndsi_compute.where(ts_ds.isel(time=0)>0)
    return scenes_ndwi_compute

def get_s2_rgb(ts_ds,cloud_cover_threshold=20):
    '''
    Returns the rgb time series of the area covered by a given xarray dataset using Sentinel 2 imagery

            Parameters:
                    ts_ds (xarray dataset): the area we will return the rgb over

            Returns:
                    scenes_rgb_compute (xarray dataset): computed rgb time series with same spatial grid and temporal bounds as as the input dataset
    '''
    # go from ds to lat lon here
    ds_4326 = ts_ds.rio.reproject('EPSG:4326', resampling=rio.enums.Resampling.cubic)
    box = shapely.geometry.box(*ds_4326.rio.bounds())
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[box])
    # must be lat lot bounding box
    lower_lon, upper_lat, upper_lon, lower_lat = bbox_gdf.bounds.values[0]
    #lower_lon, upper_lat, upper_lon, lower_lat = gdf.geometry.total_bounds

    lon = (lower_lon + upper_lon)/2
    lat = (lower_lat + upper_lat)/2
    
    start_time = pd.to_datetime(ts_ds.time[0].values).strftime('%Y-%m-%d')
    end_time = pd.to_datetime(ts_ds.time[-1].values).strftime('%Y-%m-%d')
    
    URL = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(URL)
    
    items = catalog.search(
    intersects=dict(type="Point", coordinates=[lon, lat]),
    collections=["sentinel-2-l2a"],
    datetime=f"{start_time}/{end_time}").get_all_items()
    
    string = f'{ts_ds.rio.crs}'
    epsg_code = int(string[5:])
    
    stack = stackstac.stack(items,bounds_latlon=(bbox_gdf.bounds.values[0]),epsg=epsg_code) #epsg=epsg_code
    
    if np.unique(stack['proj:epsg']).size>1:
        stack = stack[stack['proj:epsg']!=stack['epsg']]
    
    
    bounding_box_utm_gf = bbox_gdf.to_crs(stack.crs)
    xmin, ymax, xmax, ymin = bounding_box_utm_gf.bounds.values[0]

    lowcloud = stack[stack["eo:cloud_cover"] < cloud_cover_threshold]
    lowcloud = lowcloud
    #lowcloud = lowcloud.drop_duplicates("time","first")
    # snow.groupby(snow.time.dt.date).mean() use this for groupby date
    rgb = lowcloud.sel(band=["red","green","blue"])
    
    time_slice = slice(start_time,end_time)
    scenes_rgb = rgb.sel(x=slice(xmin,xmax),y=slice(ymin,ymax)).sel(time=time_slice)
    scenes_rgb = scenes_rgb.rio.write_crs(stack.rio.crs)
    scenes_rgb_compute = scenes_rgb.resample(time='1D',skipna=True).mean("time", keep_attrs=True).dropna('time',how='all')#.compute()
    
    # epsg problems?
    
    return scenes_rgb_compute

def plot_bs_ndsi_swe_precip(ts_ds,ax=None,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    if ax is None:
        ax = plt.gca()
    f = plt.gcf()    
    
    plt.style.use('seaborn-dark')

    snwd_ax = ax.twinx()
    precip_ax = ax.twinx()
    ndsi_ax = ax.twinx()
    
    #host.set_xlim(0, 2)
    #host.set_ylim(0, 2)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(1, 65)
    #snwd_ax.set_ylim(bottom=0)
    snwd_ax.set_ylim([0,480])

    
    ax.set_xlabel("Time")
    ax.set_ylabel("Backscatter [Watts]")
    snwd_ax.set_ylabel("Snow Depth / SWE [cm]")
    precip_ax.set_ylabel("Precipitation [cm]")
    ndsi_ax.set_ylabel("NDSI")

    #bs_plot1, = ax.plot(ts_ds[ts_ds.coords['sat:orbit_state']=='ascending'].time,ts_ds[ts_ds.coords['sat:orbit_state']=='ascending'].mean(dim=['x','y']),color='red',label='Backscatter (Ascending)')
    #bs_plot2, = ax.plot(ts_ds[ts_ds.coords['sat:orbit_state']=='descending'].time,ts_ds[ts_ds.coords['sat:orbit_state']=='descending'].mean(dim=['x','y']),color='orange',label='Backscatter (Descending)')
    for orbit in np.unique(ts_ds.coords['sat:relative_orbit']):
        direction = ts_ds[ts_ds.coords['sat:relative_orbit']==orbit]['sat:orbit_state'].values[0].capitalize()
        ax.plot(ts_ds[ts_ds.coords['sat:relative_orbit']==orbit].time,ts_ds[ts_ds.coords['sat:relative_orbit']==orbit].mean(dim=['x','y']),label=f'Orbit {orbit} ({direction})')

    snow = get_s2_ndsi(ts_ds)
    ndsi_plot, = ndsi_ax.plot(snow.time,snow.mean(dim=['x','y']),color='black',label='NDSI')
    snotel_snwd = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:SNWD_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    snwd_plot = snwd_ax.scatter(snotel_snwd.index,2.54*snotel_snwd.iloc[:,0],color='blueviolet',alpha=0.7,label='Snow Depth')
    
    snotel_swe = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:WTEQ_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    swe_plot = snwd_ax.scatter(snotel_swe.index,2.54*snotel_swe.iloc[:,0],color='darkturquoise',alpha=0.7,label='SWE')
    
    #print(snotel_snwd)
    #ax.scatter(x=snotel_snwd.index,y=snotel_snwd['value'],label='Snow Depth')
    snotel_precip = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:PRCPSA_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    precip_plot = precip_ax.bar(snotel_precip.index,2.54*snotel_precip.iloc[:,0],color='blue',alpha=0.4,label='Precipitation')
    lns = [ndsi_plot,snwd_plot,swe_plot,precip_plot]
    ax.legend(handles=lns,loc='best')
    
    extra_legend = ax.legend(handles=lns,loc='upper center')
    ax.legend(loc='upper right')
    ax.add_artist(extra_legend)
    #time_slice = slice('2015-01-01','2022-01-01')
    #ax.set_xlim([time_slice.start,time_slice.stop])
    
    precip_ax.spines['right'].set_position(('outward', 60))
    ndsi_ax.spines['left'].set_position(('outward', 60))
    #ndsi_ax.yaxis.label.set_position(('outward', 60))
    
    ndsi_ax.spines["left"].set_visible(True)
    ndsi_ax.yaxis.set_label_position('left') 
    ndsi_ax.yaxis.set_ticks_position('left')
    
    #ax.yaxis.label.set_color(bs_plot1.get_color())
    ndsi_ax.yaxis.label.set_color(ndsi_plot.get_color())
    snwd_ax.yaxis.label.set_color('blueviolet')
    precip_ax.yaxis.label.set_color('blue')
    
    ax.set_xlim([datetime.strptime(start_date,'%Y-%m-%d'),datetime.strptime(end_date,'%Y-%m-%d')])   
    
    ax.set_title('S1 Backscatter, S2 NDSI, SNOTEL Snow Depth, SWE, and Precipitation')
    plt.tight_layout()
    
def plot_bs_ndsi_swe_precip_with_context(ts_ds,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    
    f,ax=plt.subplots(1,2,figsize=(25,5),gridspec_kw={'width_ratios': [1, 3]})
    
    plt.style.use('seaborn-dark')

    snwd_ax = ax[1].twinx()
    precip_ax = ax[1].twinx()
    ndsi_ax = ax[1].twinx()
    
    #host.set_xlim(0, 2)
    #host.set_ylim(0, 2)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(1, 65)
    #snwd_ax.set_ylim(bottom=0)
    snwd_ax.set_ylim([0,480])

    
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Backscatter [dB]")
    snwd_ax.set_ylabel("Snow Depth / SWE [cm]")
    precip_ax.set_ylabel("Precipitation [cm]")
    ndsi_ax.set_ylabel("NDSI")

    #bs_plot1, = ax[1].plot(ts_ds[ts_ds.coords['sat:orbit_state']=='ascending'].time,ts_ds[ts_ds.coords['sat:orbit_state']=='ascending'].mean(dim=['x','y']),color='red',label='Backscatter (Ascending)')
    #bs_plot2, = ax[1].plot(ts_ds[ts_ds.coords['sat:orbit_state']=='descending'].time,ts_ds[ts_ds.coords['sat:orbit_state']=='descending'].mean(dim=['x','y']),color='orange',label='Backscatter (Descending)')

    for orbit in np.unique(ts_ds.coords['sat:relative_orbit']):
        direction = ts_ds[ts_ds.coords['sat:relative_orbit']==orbit]['sat:orbit_state'].values[0].capitalize()
        ax[1].plot(ts_ds[ts_ds.coords['sat:relative_orbit']==orbit].time,10.0*np.log10(ts_ds[ts_ds.coords['sat:relative_orbit']==orbit].mean(dim=['x','y'])),label=f'Orbit {orbit} ({direction})')
    #ax[1].legend()
    
    
    snow = get_s2_ndsi(ts_ds)
    ndsi_plot, = ndsi_ax.plot(snow.time,snow.mean(dim=['x','y']),color='black',label='NDSI')
    snotel_snwd = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:SNWD_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    snwd_plot = snwd_ax.scatter(snotel_snwd.index,2.54*snotel_snwd.iloc[:,0],color='blueviolet',alpha=0.7,label='Snow Depth')
    
    snotel_swe = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:WTEQ_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    swe_plot = snwd_ax.scatter(snotel_swe.index,2.54*snotel_swe.iloc[:,0],color='darkturquoise',alpha=0.7,label='SWE')
    
    #print(snotel_snwd)
    #ax.scatter(x=snotel_snwd.index,y=snotel_snwd['value'],label='Snow Depth')
    snotel_precip = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:PRCPSA_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    snotel_temp = get_closest_snotel_data(ts_ds,variable_code='SNOTEL:TAVG_D',distance_cutoff=30,closest=True,start_date='1900-01-01', end_date=datetime.today().strftime('%Y-%m-%d'))
    snotel_temp=(snotel_temp-32)/1.8
    temp_precip_gdf = pd.concat([snotel_temp,snotel_precip],axis=1,join='inner')
    temp_precip_gdf.set_axis(['Temperature','Precip'],axis=1,inplace=True)
    conditions = [(temp_precip_gdf['Temperature'] <=0),(temp_precip_gdf['Temperature'] > 0)]
    values = ['lightcoral', 'blue']
    temp_precip_gdf['bar_color'] = np.select(conditions, values)
    
    
    precip_plot = precip_ax.bar(temp_precip_gdf.index,2.54*temp_precip_gdf.iloc[:,1],color=temp_precip_gdf['bar_color'],alpha=0.4,label='Precipitation')
    lns = [ndsi_plot,snwd_plot,swe_plot,precip_plot]
    extra_legend = ax[1].legend(handles=lns,loc='upper center')
    ax[1].legend(loc='upper right')
    ax[1].add_artist(extra_legend)
    #time_slice = slice('2015-01-01','2022-01-01')
    #ax.set_xlim([time_slice.start,time_slice.stop])
    
    precip_ax.spines['right'].set_position(('outward', 60))
    ndsi_ax.spines['left'].set_position(('outward', 60))
    #ndsi_ax.yaxis.label.set_position(('outward', 60))
    
    ndsi_ax.spines["left"].set_visible(True)
    ndsi_ax.yaxis.set_label_position('left') 
    ndsi_ax.yaxis.set_ticks_position('left')
    
    #ax[1].yaxis.label.set_color(bs_plot1.get_color())
    ndsi_ax.yaxis.label.set_color(ndsi_plot.get_color())
    snwd_ax.yaxis.label.set_color('blueviolet')
    precip_ax.yaxis.label.set_color('blue')
    
    ax[1].set_xlim([datetime.strptime(start_date,'%Y-%m-%d'),datetime.strptime(end_date,'%Y-%m-%d')])   
    
    plt.tight_layout()
    
    
    sites_gdf = find_closest_snotel(ts_ds)
    sites_gdf[sites_gdf['distance_km']==sites_gdf['distance_km'].min()].plot(ax=ax[0],color='red',marker='*')
    
    for x, y, label1, label2, label3, label4 in zip(sites_gdf.geometry.x, sites_gdf.geometry.y, sites_gdf.name, sites_gdf.code, sites_gdf.distance_km, sites_gdf.elevation_m):
        ax[0].annotate(f'{label1} \n{label2} \nElevation:{label4:.0f} m \nProximity:{label3:.2f} km', xy=(x, y), xytext=(15, -30), textcoords="offset points", fontsize=10,bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'))
        break
    
    minx, miny, maxx, maxy = ts_ds.rio.bounds()
    distance_cutoff=1

    
    #ts_ds.isel(time=0).plot(ax=ax[0],vmax=1.0,cmap='gray',add_colorbar=False)
    get_runoff_onset(ts_ds).dt.dayofyear.plot(ax=ax[0],cmap='twilight')

    
    ax[0].set_xlim([minx-1000*distance_cutoff*1.2,maxx+1000*distance_cutoff*1.2])
    ax[0].set_ylim([miny-1000*distance_cutoff*1.2,maxy+1000*distance_cutoff*1.2])
    
    ctx.add_basemap(ax=ax[0], crs=ts_ds.rio.crs, source=ctx.providers.Stamen.Terrain)
    ax[0].set_title('Area of Interest')
    

    
    site_name = sites_gdf[sites_gdf['distance_km']==sites_gdf['distance_km'].min()]['code'].values[0]
    ax[1].set_title(f'S1 Backscatter, S2 NDSI, {site_name} Snow Depth, SWE, and Precipitation')
