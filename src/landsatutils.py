'''
This file is created by Emma Liu (liuwj@stanford.edu)
for CME 215 class project.

This codes contains utility functions that process
landsat data to calculate NDVI

Examples to use these functions can be found in:
wildfire_dataprocessing.ipynb
'''

from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import pandas as pd
from dask_geopandas import read_file
from tabulate import tabulate
import tarfile
import tifffile as tiff

from shapely.geometry import box
import shapely
import geopandas as gpd
from pyproj import Transformer
from rasterio.mask import mask
import rasterio
from scipy.ndimage import zoom

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def search_landsat_scene(api,lat,lon,m_start_date,m_end_date):
    # https://github.com/yannforget/landsatxplore
    scenes = api.search(
        dataset='landsat_ot_c2_l2',
        latitude= lat,
        longitude = lon,
        start_date=m_start_date,
        end_date=m_end_date,
        max_cloud_cover=50
    )

    # If there are scenes available, download the first scene
    if scenes:
        print("Found ",len(scenes), " scenes within the requested time period at lat: ",lat, " lon: ",lon)
    else:
        print("No Landsat scenes found for the specified area and time period.")

    # Create a DataFrame from the scenes
    df_scenes = pd.DataFrame(scenes)
    df_scenes = df_scenes[['display_id','corner_lower_left_latitude', 'corner_lower_left_longitude',
                          'corner_upper_right_latitude', 'corner_upper_right_longitude','satellite','cloud_cover','acquisition_date']]
    df_scenes.sort_values('acquisition_date', ascending=False, inplace=True)

    print(tabulate(df_scenes, headers='keys', tablefmt='psql'))

def download_landsat_scene(ID_pre,ID_post,ee):
    # Download the scene
    try:
        ee.download(ID_pre, output_dir='./data')
        print('{} succesful'.format(ID_pre))
    # Additional error handling
    except:
        if os.path.isfile('./data/{}.tar'.format(ID_pre)):
            print('{} error but file exists'.format(ID_pre))
        else:
            print('{} error'.format(ID_pre))
    # Download the scene
    try:
        ee.download(ID_post, output_dir='./data')
        print('{} succesful'.format(ID_post))
    # Additional error handling
    except:
        if os.path.isfile('./data/{}.tar'.format(ID_post)):
            print('{} error but file exists'.format(ID_post))
        else:
            print('{} error'.format(ID_post))
    
    # Extract files from tar archive
    tar = tarfile.open('./data/{}.tar'.format(ID_pre))
    tar.extractall('./data/{}'.format(ID_pre))
    tar = tarfile.open('./data/{}.tar'.format(ID_post))
    tar.extractall('./data/{}'.format(ID_post))
    tar.close()

def visualize_scene(ID_pre,ID_post,m_start_date,m_end_date):
    # Load Blue (B2), Green (B3) and Red (B4) bands
    B2 = tiff.imread('./data/{}/{}_SR_B2.TIF'.format(ID_pre, ID_pre))
    B3 = tiff.imread('./data/{}/{}_SR_B3.TIF'.format(ID_pre, ID_pre))
    B4 = tiff.imread('./data/{}/{}_SR_B4.TIF'.format(ID_pre, ID_pre))
    # Stack and scale bands
    RGB_pre = np.dstack((B4, B3, B2))
    RGB_pre = np.clip(RGB_pre*0.0000275-0.2, 0, 1)
    # Clip to enhance contrast
    RGB_pre = np.clip(RGB_pre,0,0.2)/0.2

    # Load Blue (B2), Green (B3) and Red (B4) bands
    B2 = tiff.imread('./data/{}/{}_SR_B2.TIF'.format(ID_post, ID_post))
    B3 = tiff.imread('./data/{}/{}_SR_B3.TIF'.format(ID_post, ID_post))
    B4 = tiff.imread('./data/{}/{}_SR_B4.TIF'.format(ID_post, ID_post))
    # Stack and scale bands
    RGB_post = np.dstack((B4, B3, B2))
    RGB_post = np.clip(RGB_post*0.0000275-0.2, 0, 1)
    # Clip to enhance contrast
    RGB_post = np.clip(RGB_post,0,0.2)/0.2

    figure, ax = plt.subplots(1, 2, figsize=(10, 10))
    im0 = ax[0].imshow(np.flipud(RGB_pre),origin='lower')
    ax[0].set_title('Satellite '+m_start_date)
    im1 = ax[1].imshow(np.flipud(RGB_post),origin='lower')
    ax[1].set_title('Satellite '+m_end_date)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.show()

def read_band(band_ID, ID, coord_box_df=None):
    with rasterio.open("./data/{}/{}_SR_B{}.TIF".format(ID, ID, band_ID)) as band:
        # Get the bounding box of the raster
        raster_bbox = box(*band.bounds)
        print(band.bounds)
        print(f"Raster CRS: {band.crs}")
        # Check if a dataframe of coordinate boxes is provided
        if coord_box_df is not None:
            # Ensure coord_box_df is a GeoDataFrame and has a 'geometry' column
            if not isinstance(coord_box_df, gpd.GeoDataFrame) or 'geometry' not in coord_box_df.columns:
                raise ValueError("coord_box_df must be a GeoDataFrame with a 'geometry' column.")

            # Filter geometries in coord_box_df that intersect with the raster's bounding box
            intersecting_geometries = coord_box_df[coord_box_df.intersects(raster_bbox)]
        else:
            # If no coord_box_df is provided, create a GeoDataFrame with the raster's bounding box as its only geometry
            # This implies considering the entire raster's bounding box as the area of interest
            intersecting_geometries = gpd.GeoDataFrame({'geometry': [raster_bbox]}, crs=band.crs)

        if not intersecting_geometries.empty:
            band_meta = band.meta
            # Apply the mask
            band_arr, band_transform = mask(band, intersecting_geometries["geometry"], crop=True)
            # select just the 2-D array (by default a 3-D array is returned even though we only have one band)
            band_arr = band_arr[0,:,:]
        else:
            print("No geometries intersect with the raster's extent. Skipping masking.")
            # Handle the case when no geometries intersect with the raster
            # For example, you can set shipc_arr and shipc_transform to None or use a default value
            band_arr = None
            band_transform = None

    # print(raster_bbox)
    # print(coord_box_df)
    band_meta.update({"driver": "GTiff",
                 "height": band_arr.shape[0],
                 "width": band_arr.shape[1],
                 "transform": band_transform,
                 "compress": "lzw"})
    return band_arr, band_meta

def get_bounding_box_from_fire(fire_id):
    # read in fire data. This is concat_speed
    file_path = "/content/drive/MyDrive/stanford/courses/CME 215 ML and Physical Processes/derived_data/fireID_"+str(fire_id)+"_fire_event_computed.gpkg"
    gdf = read_file(file_path, npartitions=10)
    
    concat_speed = gdf.compute().to_crs(epsg=4326)

    min_x = concat_speed.bounds.minx.min()
    max_x = concat_speed.bounds.maxx.max()
    min_y = concat_speed.bounds.miny.min()
    max_y = concat_speed.bounds.maxy.max()
    return min_x, min_y, max_x, max_y

def read_band_data(ID,coord_box_df):
    band_1, band_meta_1 = read_band(1, ID, coord_box_df=coord_box_df)
    band_2, band_meta_2 = read_band(2, ID, coord_box_df=coord_box_df)
    band_3, band_meta_3 = read_band(3, ID, coord_box_df=coord_box_df)
    band_4, band_meta_4 = read_band(4, ID, coord_box_df=coord_box_df)
    band_5, band_meta_5 = read_band(5, ID, coord_box_df=coord_box_df)
    band_6, band_meta_6 = read_band(6, ID, coord_box_df=coord_box_df)
    band_7, band_meta_7 = read_band(7, ID, coord_box_df=coord_box_df)
    return band_1,band_meta_1,band_2,band_meta_2,band_3,band_meta_3,band_4,band_meta_4,band_5,band_meta_5,band_6,band_meta_6,band_7,band_meta_7

def compute_NDVI(band_4,band_5):
    NDVI = (band_5 - band_4)/(band_5 + band_4+1)
    NDVI[np.abs(NDVI)>1]=1.0
    return NDVI

def visualize_NDVI(NDVI_pre,NDVI_pos,m_start_date,m_end_date,fire_id):
    figure, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plotting NDVI_pre
    im0 = ax[0].imshow(np.flipud(NDVI_pre), origin='lower', cmap='gist_earth_r')
    ax[0].set_title('Fire ID:'+str(fire_id)+' NDVI ' + m_start_date)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = figure.colorbar(im0, cax=cax0)
    # cbar0.ax.set_ylabel('NDVI Values')

    # Plotting NDVI_pos
    im1 = ax[1].imshow(np.flipud(NDVI_pos), origin='lower', cmap='gist_earth_r')
    ax[1].set_title('Fire ID:'+str(fire_id)+' NDVI ' + m_end_date)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = figure.colorbar(im1, cax=cax1)
    # cbar1.ax.set_ylabel('NDVI Values')
    plt.show()

def resample_NDVI(NDVI, grid_resolution):
    original_shape = NDVI.shape
    target_shape = (grid_resolution, grid_resolution)

    zoom_factors = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])

    # Resize the data using zoom factors
    # 'order' parameter controls the order of the spline interpolation (0=nearest, 1=bilinear, 3=cubic, etc.)
    resampled_NDVI = zoom(NDVI, zoom_factors, order=3)

    print("Original shape:", original_shape)
    print("Resampled shape:", resampled_NDVI.shape)
    return resampled_NDVI