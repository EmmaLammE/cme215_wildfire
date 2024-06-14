'''
This file is created by Emma Liu (liuwj@stanford.edu)
for CME 215 class project.

This codes contains utility functions that process
rate of spread data

Examples to use these functions can be found in:
wildfire_dataprocessing.ipynb
'''

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
from dask_geopandas import read_file
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import contextily as cx
import pyproj
import numpy as np
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import imageio
import os

class StadiaStamen(cimgt.Stamen):
    def _image_url(self, tile):
         x,y,z = tile
         url = f"https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.jpg?api_key=c19b18d7-14f8-45e7-b02b-529cdd22b80e"
         return url


def create_fire_progression_gif(fire_event_computed,save_path):
    # Calculate the overall min and max x and y coordinates
    min_x = fire_event_computed.bounds.minx.min()
    max_x = fire_event_computed.bounds.maxx.max()
    min_y = fire_event_computed.bounds.miny.min()
    max_y = fire_event_computed.bounds.maxy.max()
    print(f"Overall min and max coordinates: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Create a directory to store the images
    os.makedirs('fire_event_images', exist_ok=True)

    # Generate plots for each time slice and save as images
    filenames = []
    for i in range(len(fire_event_computed)):
        time_slice = fire_event_computed.iloc[i]
        time_slice_geometry = gpd.GeoSeries(time_slice['geometry'])
        
        # Plot the geometry
        fig, ax = plt.subplots(figsize=(10, 10))
        time_slice_geometry.plot(ax=ax)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        # plt.title(f"Geometry of Fire Event ID {time_slice['id']} - Time Slice {i + 1}")
        plt.title(f"Fire ID {time_slice['id']}, {time_slice['eco_name']}\n dur {i + 1}/{time_slice['event_dur']}, tot area {time_slice['tot_ar_km2']:.2f}")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Save the plot as an image
        filename = f'fire_event_images/fire_event_{i + 1}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Create a GIF from the images
    with imageio.get_writer(save_path+'_animation.gif', mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up by removing the individual image files
    for filename in filenames:
        os.remove(filename)


def create_fire_progression_gif_cartopy(fire_event_computed, save_path):
    # Re-project the entire dataset to WGS 84 (latitude and longitude)
    fire_event_computed = fire_event_computed.to_crs(epsg=4326)
    # Calculate the overall min and max x and y coordinates
    min_x = fire_event_computed.bounds.minx.min()
    max_x = fire_event_computed.bounds.maxx.max()
    min_y = fire_event_computed.bounds.miny.min()
    max_y = fire_event_computed.bounds.maxy.max()
    print(f"Overall min and max coordinates: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Create a directory to store the images
    os.makedirs('fire_event_images', exist_ok=True)

    # Generate plots for each time slice and save as images
    filenames = []
    for i in range(len(fire_event_computed)):
        time_slice = fire_event_computed.iloc[i:i+1]  # Select the ith slice and keep it as a DataFrame
        
        # Plot the geometry
        fig, ax = plt.subplots(figsize=(7, 7))
        time_slice.plot(ax=ax, color='tab:red', edgecolor='black', alpha=0.5)
        # Set the extent to the bounds of the overall geometry before adding the basemap
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        # Add the basemap
        cx.add_basemap(ax, crs=time_slice.crs, source='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png?api_key=c19b18d7-14f8-45e7-b02b-529cdd22b80e')

        # Add gridlines (optional)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set title and labels
        ax.set_title(f"Fire ID {time_slice['id'].values[0]}, {time_slice['eco_name'].values[0]}\nDuration: {i + 1}/{time_slice['event_dur'].values[0]}, Total Area: {time_slice['tot_ar_km2'].values[0]:.2f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Save the plot as an image
        filename = f'fire_event_images/fire_event_{i + 1}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Create a GIF from the images
    with imageio.get_writer(save_path + '_animation.gif', mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up by removing the individual image files
    for filename in filenames:
        os.remove(filename)

# def create_fire_spread_rate_gif_cartopy(accumulated_speed, save_path):

def create_cum_fire_progression_gif_cartopy(fire_event_computed, save_path):
    # Re-project the entire dataset to WGS 84 (latitude and longitude)
    fire_event_computed = fire_event_computed.to_crs(epsg=4326)
    
    # Calculate the overall min and max x and y coordinates after re-projection
    min_x = fire_event_computed.bounds.minx.min()
    max_x = fire_event_computed.bounds.maxx.max()
    min_y = fire_event_computed.bounds.miny.min()
    max_y = fire_event_computed.bounds.maxy.max()
    print(f"Overall min and max coordinates: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Initialize an empty GeoDataFrame to accumulate geometries
    accumulated_geometries = gpd.GeoDataFrame(columns=fire_event_computed.columns, crs=fire_event_computed.crs)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    def update(i):
        ax.clear()
        
        # Accumulate the geometries
        time_slice = fire_event_computed.iloc[i:i+1]
        nonlocal accumulated_geometries
        accumulated_geometries = pd.concat([accumulated_geometries, time_slice], ignore_index=True)
        
        # Plot the accumulated geometries
        accumulated_geometries.plot(ax=ax, color='tab:red', edgecolor='none', alpha=0.5)

        # Set the extent to the bounds of the overall geometry before adding the basemap
        ax.set_xlim(min_x-0.05, max_x+0.05)
        ax.set_ylim(min_y-0.05, max_y+0.05)

        # Add the basemap
        cx.add_basemap(ax, crs=time_slice.crs, source='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png?api_key=c19b18d7-14f8-45e7-b02b-529cdd22b80e')

        # Convert ignition point coordinates to latitude and longitude
        sinusoidal_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
        wgs84_proj = "epsg:4326"
        transformer = pyproj.Transformer.from_crs(sinusoidal_proj, wgs84_proj, always_xy=True)
        lon, lat = transformer.transform(fire_event_computed['ig_utm_x'].iloc[0], fire_event_computed['ig_utm_y'].iloc[0])
        
        # Plot the ignition point as a fire symbol
        ax.scatter(lon, lat, color='yellow', marker='*', s=200, label='Ignition Point', edgecolors='black')

        # Add gridlines (optional)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set title and labels
        ax.set_title(f"Fire ID {time_slice['id'].values[0]}, {time_slice['eco_name'].values[0]}\nDuration: {i + 1}/{time_slice['event_dur'].values[0]}, Total Area: {time_slice['tot_ar_km2'].values[0]:.2f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ani = FuncAnimation(fig, update, frames=len(fire_event_computed), repeat=False)
    
    # Save the animation as a GIF
    gif_path = save_path + '_animation.gif'
    ani.save(gif_path, writer=PillowWriter(fps=10)) # change fps to speed up
    print("GIF saved as:", gif_path)

    return accumulated_geometries

def get_accumulated_fire_geometry(fire_event_computed):
    # Re-project the entire dataset to WGS 84 (latitude and longitude)
    fire_event_computed = fire_event_computed.to_crs(epsg=4326)
    for i in range(len(fire_event_computed)):
        time_slice = fire_event_computed.iloc[i:i+1]  # Select the ith slice and keep it as a DataFrame
        accumulated_geometries = pd.concat([accumulated_geometries, time_slice], ignore_index=True)
    return accumulated_geometries

def plot_daily_fire_slice(fire_event_computed,slice):
    first_large_fire = fire_event_computed.iloc[slice]
    first_large_fire_geometry = gpd.GeoSeries(first_large_fire['geometry'])
    fire_event_id = first_large_fire['id']
    # Plot the geometry
    first_large_fire_geometry.plot()
    plt.title(f"Fire ID {first_large_fire['id']}, {first_large_fire['eco_name']}\n dur {first_large_fire['event_dur']}, tot area {first_large_fire['tot_ar_km2']:.2f}")
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def plot_one_fire_event_final_peri(fire_event_computed,fire_id):
    fire_event = fire_event_computed[fire_event_computed['id'] == fire_id]
    # Re-project the geometry to WGS 84 (latitude and longitude)
    # fire_event = fire_event.to_crs(epsg=4326)
    # Extract the re-projected geometry
    fire_geometry = gpd.GeoSeries(fire_event['geometry'])
    # Plot the geometry
    fire_geometry.plot()
    fire_id_value = fire_event['id'].iloc[0]
    eco_name_value = fire_event['eco_name'].iloc[0]
    event_dur_value = fire_event['event_dur'].iloc[0]
    tot_ar_km2_value = fire_event['tot_ar_km2'].iloc[0]

    # Set the title with formatted values
    plt.title(f"Fire ID {fire_id_value}, {eco_name_value}\n dur {event_dur_value}, tot area {tot_ar_km2_value:.2f}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def plot_one_fire_event_final_peri_cartopy(fire_event_computed, fire_id):
    # Select the specific fire event to plot
    fire_event = fire_event_computed[fire_event_computed['id'] == fire_id]

    # Re-project the geometry to WGS 84 (latitude and longitude)
    fire_event = fire_event.to_crs(epsg=4326)

    # Extract the re-projected geometry
    fire_event_geometry = gpd.GeoSeries(fire_event['geometry'])

    # Extract fire event details
    fire_id_value = fire_event['id'].iloc[0]
    eco_name_value = fire_event['eco_name'].iloc[0]
    event_dur_value = fire_event['event_dur'].iloc[0]
    tot_ar_km2_value = fire_event['tot_ar_km2'].iloc[0]
    
    # Plot the geometry
    ax = fire_event.plot(figsize=(7,7),color='tab:red', edgecolor='black', alpha=0.5)

    # Add the basemap
    # print(fire_event.crs)
    cx.add_basemap(ax, crs=fire_event.crs, source='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png?api_key=c19b18d7-14f8-45e7-b02b-529cdd22b80e')
    # Set the extent to the bounds of the fire event geometry
    minx, miny, maxx, maxy = fire_event_geometry.total_bounds
    ax.set_xlim(minx-0.05, maxx+0.05)
    ax.set_ylim(miny-0.05, maxy+0.05)

    # Convert ignition point coordinates to latitude and longitude
    sinusoidal_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    wgs84_proj = "epsg:4326"
    transformer = pyproj.Transformer.from_crs(sinusoidal_proj, wgs84_proj, always_xy=True)
    lon, lat = transformer.transform(fire_event['ig_utm_x'].iloc[0], fire_event['ig_utm_y'].iloc[0])
    # Plot the ignition point as a fire symbol
    plt.scatter(lon, lat, color='yellow', marker='*', s=200, label='Ignition Point', edgecolors='black')
  
    # # Add map features for better context
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.LAND, edgecolor='black')
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    # Add gridlines (optional)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set title and labels
    ax.set_title(f"Fire ID {fire_id_value}, {eco_name_value}\nDuration: {event_dur_value}, Total Area: {tot_ar_km2_value:.2f}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()

def arrange_by_dur_and_save(fires, save_path):
    # Compute the filtered data (this will load the necessary data into memory)
    fires_computed = fires.compute()

    # Convert to GeoPandas DataFrame
    fires_computed = gpd.GeoDataFrame(fires_computed, geometry='geometry')

    # Define the sinusoidal projection (MODIS) and WGS 84 projection
    sinusoidal_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    wgs84_proj = "epsg:4326"

    # Create a transformer object
    transformer = pyproj.Transformer.from_crs(sinusoidal_proj, wgs84_proj, always_xy=True)

    # Convert x, y coordinates to latitude and longitude
    def convert_to_lat_lon(row):
        lon, lat = transformer.transform(row['ig_utm_x'], row['ig_utm_y'])
        return lat, lon

    fires_computed[['latitude', 'longitude']] = fires_computed.apply(lambda row: convert_to_lat_lon(row), axis=1, result_type='expand')

    # Sort the DataFrame by event duration in descending order
    sorted_fires = fires_computed.sort_values(by='event_dur', ascending=False)

    # Print the number of fire events
    print(f"Number of fire events: {len(sorted_fires)}")

    # Print summary of the first few rows
    print(sorted_fires[['id', 'ig_date', 'latitude', 'longitude', 'event_dur', 'tot_ar_km2']].head())

    # Save the sorted summary to CSV
    sorted_fires.to_csv(save_path + "fires_sorted_by_event_dur.csv", index=False)

    return sorted_fires

def arrange_by_durArea_and_save(fires, save_path, weight_dur=1, weight_area=1):
    # Compute the filtered data (this will load the necessary data into memory)
    fires_computed = fires.compute()

    # Convert to GeoPandas DataFrame
    fires_computed = gpd.GeoDataFrame(fires_computed, geometry='geometry')

    # Define the sinusoidal projection (MODIS) and WGS 84 projection
    sinusoidal_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    wgs84_proj = "epsg:4326"

    # Create a transformer object
    transformer = pyproj.Transformer.from_crs(sinusoidal_proj, wgs84_proj, always_xy=True)

    # Convert x, y coordinates to latitude and longitude
    def convert_to_lat_lon(row):
        lon, lat = transformer.transform(row['ig_utm_x'], row['ig_utm_y'])
        return lat, lon

    fires_computed[['latitude', 'longitude']] = fires_computed.apply(lambda row: convert_to_lat_lon(row), axis=1, result_type='expand')

    # Create a combined score based on the weighted sum of duration and area
    fires_computed['combined_score'] = (weight_dur * fires_computed['event_dur']) + (weight_area * fires_computed['tot_ar_km2'])

    # Sort the DataFrame by the combined score in descending order
    sorted_fires = fires_computed.sort_values(by='combined_score', ascending=False)

    # Print the number of fire events
    print(f"Number of fire events: {len(sorted_fires)}")

    # Print summary of the first few rows
    print(sorted_fires[['id', 'ig_date', 'latitude', 'longitude', 'event_dur', 'tot_ar_km2', 'combined_score']].head())

    # Save the sorted summary to CSV
    sorted_fires.to_csv(save_path + "fires_sorted_by_dur_area.csv", index=False)

    return sorted_fires


def extract_one_fire_event(gdf,fire_id):
    fire_event = gdf[gdf['id'] == fire_id]

    # Compute the filtered data (this will load the necessary data into memory)
    fire_event_computed = fire_event.compute()
    # Convert to GeoPandas DataFrame
    fire_event_computed = gpd.GeoDataFrame(fire_event_computed, geometry='geometry')

    # Print the number of occurrences of the fire event ID
    print(f"Number of occurrences of fire event ID {fire_id}: {len(fire_event_computed)}")

    # Print summary of the first few rows
    print(fire_event_computed[['id', 'ig_date', 'ig_utm_x', 'ig_utm_y','event_dur', 'tot_ar_km2']].head())

    # Save summary to CSV if needed
    fire_event_computed.to_csv("fire_event_"+str(fire_id)+"_summary.csv", index=False)

    return fire_event_computed

def calculate_grid_resolution(fire_geometry,grid_resolution,scaling=1.0):
    # this fire_event_computed should be before crs transfer to 4326 (lat long)
    # i.e. the max min x y should be in the original utm
    # such that the computed dx dy are in meters
    min_x = fire_geometry.bounds[0]
    max_x = fire_geometry.bounds[2]
    min_y = fire_geometry.bounds[1]
    max_y = fire_geometry.bounds[3]
    dx = (max_x - min_x) / (grid_resolution - 1)*scaling
    dy = (max_y - min_y) / (grid_resolution - 1)*scaling
    print("   Grid info: \n    x: [",max_x,",",min_x,"], dx = ",dx,
        "\n    y: [",max_y,",",min_y,"], dy = ",dy,"\n","   scaling = ",scaling)
    return dx,dy

def get_both_gridded_fire_area(previous_fire_front,current_fire_front,grid_resolution):
    grid_points_pre = []
    grid_points_cur = []
    for x in np.linspace(previous_fire_front.bounds[0], previous_fire_front.bounds[2], grid_resolution):
        for y in np.linspace(previous_fire_front.bounds[1], previous_fire_front.bounds[3], grid_resolution):
            point = Point(x, y)
            if previous_fire_front.contains(point):
                grid_points_pre.append(point)
            if current_fire_front.contains(point):
                grid_points_cur.append(point)
    print("num of grid_points_pre:", len(grid_points_pre))
    print("num of grid_points_cur:", len(grid_points_cur))
    # Create a GeoDataFrame with grid points, speeds, and directions
    grid_pre = gpd.GeoDataFrame(geometry=grid_points_pre)
    grid_cur = gpd.GeoDataFrame(geometry=grid_points_cur)
    return grid_pre, grid_cur

def get_one_gridded_fire_area(fire_event_computed,grid_resolution, crs):
    fire_geometry_4326 = fire_event_computed.to_crs(epsg=4326)
    min_x = fire_geometry_4326.bounds.minx.min()
    max_x = fire_geometry_4326.bounds.maxx.max()
    min_y = fire_geometry_4326.bounds.miny.min()
    max_y = fire_geometry_4326.bounds.maxy.max()
    print("Grid info, x [",min_x,",",max_x,"], dx", (max_x-min_x)/(grid_resolution-1))
    print("Grid info, y [",min_y,",",max_y,"], dy", (max_y-min_y)/(grid_resolution-1))

    grid_points = []
    iter = 0
    for x in np.linspace(min_x, max_x, grid_resolution):
        print(iter)
        for y in np.linspace(min_y, max_y, grid_resolution):
            point = Point(x, y)
            for i in range(len(fire_geometry_4326)):
                if fire_geometry_4326.iloc[i].geometry.intersects(point):
                    grid_points.append(point)
                    # print(fire_event_computed.iloc[i].geometry.intersects(point))
        iter +=1
    grid = gpd.GeoDataFrame(geometry=grid_points)
    # Set CRS to match fire_front
    grid.set_crs(crs, inplace=True)
    return grid

def get_gridded_fire_clustered_points(grid,dx,dy):
    coords = np.array([[point.x, point.y] for point in grid.geometry])
    # Adjust eps and min_samples as needed
    # smaller eps means smaller distance as considered as 1 cluster, result in finer clusters
    dbscan = DBSCAN(eps=np.max((dx,dy))*1.1, min_samples=2) 
    cluster_labels = dbscan.fit_predict(coords)
    grid['cluster'] = cluster_labels
    return grid, cluster_labels

def compute_spread_speed_snapshot(grid_cur,cluster_labels,dx,dy):
    for cluster_id in np.unique(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = grid_cur[grid_cur['cluster'] == cluster_id]
            grid_cur.loc[grid_cur['cluster'] == cluster_id, 'speed'] = \
            len(cluster_points)*dx*dy/24/3600
    # Ensure the CRS is set
    # grid_cur.set_crs(grid_pre.crs, inplace=True)
    return grid_cur

def compute_spread_speed_alltimesteps(fire_event_computed,grid,grid_resolution=200):
    fire_geometry_4326 = fire_event_computed.to_crs(epsg=4326)
    # grid = get_one_gridded_fire_area(fire_event_computed,grid_resolution, fire_geometry_4326.crs)
    # print("finished constructing grid")
    min_x = fire_event_computed.bounds.minx.min()
    max_x = fire_event_computed.bounds.maxx.max()
    min_y = fire_event_computed.bounds.miny.min()
    max_y = fire_event_computed.bounds.maxy.max()
    dx = (max_x - min_x) / (grid_resolution - 1)
    dy = (max_y - min_y) / (grid_resolution - 1)
    min_x = fire_geometry_4326.bounds.minx.min()
    max_x = fire_geometry_4326.bounds.maxx.max()
    min_y = fire_geometry_4326.bounds.miny.min()
    max_y = fire_geometry_4326.bounds.maxy.max()
    dx_4326 = (max_x - min_x) / (grid_resolution - 1)
    dy_4326 = (max_y - min_y) / (grid_resolution - 1)

    concat_speed = gpd.GeoDataFrame(columns=grid.columns, crs=fire_event_computed.crs)
    spread_speed_list = []
    for i in range(len(fire_event_computed)-1):#
        current_grid_points = []
        current_fire_front = fire_geometry_4326.iloc[i+1].geometry
        for point in grid.geometry:
            if fire_geometry_4326.iloc[i+1].geometry.intersects(point):
              current_grid_points.append(point)
              #  print("adding point ", point)
        print("time: ", i, ", length of current grid:",len(current_grid_points))
        current_grid = gpd.GeoDataFrame(geometry=current_grid_points)
        current_grid.set_crs(fire_event_computed.crs, inplace=True)
        current_grid, current_cluster_labels = get_gridded_fire_clustered_points(current_grid,dx_4326,dy_4326)
        
        current_grid['speed'] = np.nan
        current_grid = compute_spread_speed_snapshot(current_grid,current_cluster_labels,dx,dy)

        concat_speed = pd.concat([concat_speed, current_grid], ignore_index=True)
        spread_speed_list.append(current_grid)
        
        # print(" num of grid_points at this step:", len(current_grid))
        # print(accumulated_speed)
    print("after combing all time steps, accumulated_speed: ",spread_speed_list)
    return spread_speed_list, concat_speed


    # # convert to lat long
    # fire_geometry = fire_event_computed.to_crs(epsg=4326)
    # fire_geometry_utm = fire_event_computed
    # fire_front = fire_geometry.iloc[0].geometry
    # fire_front_utm = fire_geometry_utm.iloc[0].geometry
    # dx, dy = calculate_grid_resolution(fire_front_utm,grid_resolution)
    # grid_pre = get_one_gridded_fire_area(fire_front,grid_resolution,fire_geometry.crs)
    
    # concat_speed = gpd.GeoDataFrame(columns=grid_pre.columns, crs=grid_pre.crs)
    # spread_speed_list = []
    # for i in range(len(fire_geometry)-1):
    #     print("calculating spread speed at time step: ", i)
    #     current_fire_front = fire_geometry.iloc[i+1].geometry
    #     current_fire_front_utm = fire_geometry_utm.iloc[i+1].geometry

    #     dx, dy = calculate_grid_resolution(current_fire_front_utm,grid_resolution)
        
    #     grid_cur = get_one_gridded_fire_area(current_fire_front,grid_resolution,fire_geometry.crs)
    #     grid_cur, cluster_labels = get_gridded_fire_clustered_points(grid_cur,dx,dy)
    #     grid_cur['speed'] = np.nan
    #     grid_cur = compute_spread_speed_snapshot(grid_pre,grid_cur,cluster_labels,dx,dy)

    #     concat_speed = pd.concat([concat_speed, grid_cur], ignore_index=True)
    #     spread_speed_list.append(grid_cur)
    #     grid_pre = grid_cur
    #     print(" num of grid_points at this step:", len(grid_pre))
    #     # print(accumulated_speed)
    # print("after combing all time steps, accumulated_speed: ",spread_speed_list)
    # return spread_speed_list, concat_speed

def plot_propagation_speed(propagation_speed_df):
    # Plot the propagation speed
    plt.figure(figsize=(10, 6))
    plt.plot(propagation_speed_df['Day'], propagation_speed_df['Propagation Speed (sq meters/day)'], marker='o', linestyle='-')
    plt.title('Fire Propagation Speed Over Time')
    plt.xlabel('Day')
    plt.ylabel('Propagation Speed (sq meters/day)')
    plt.grid(True)
    plt.show()

def plot_cluster_snapshot(grid):
    # input should be geo dataframe of 1 time step
    cluster_labels = grid['cluster']
    # Create a 2D scatter plot of grid points colored by cluster
    fig, ax = plt.subplots(figsize=(6, 6))
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Noise points
            color = 'gray'
            label = 'Noise'
        else:
            color = plt.cm.Spectral(cluster_id / len(np.unique(cluster_labels)))
            label = f'Cluster {cluster_id}'
        cluster_points = grid[grid['cluster'] == cluster_id]
        cluster_points.plot(ax=ax, marker='o', markersize=2, color=color, label=label)
    ax.set_title('Grid Points Colored by Cluster')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.show()

def create_cum_fire_spreadspeed_gif_cartopy(concat_speed, spread_speed_list, fire_event_computed, save_path):
    # Calculate the overall min and max x and y coordinates after re-projection
    min_x = concat_speed.bounds.minx.min()
    max_x = concat_speed.bounds.maxx.max()
    min_y = concat_speed.bounds.miny.min()
    max_y = concat_speed.bounds.miny.max()
    print(f"Overall min and max coordinates: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Initialize an empty GeoDataFrame to accumulate geometries
    accumulated_geometries = gpd.GeoDataFrame(columns=spread_speed_list[0].columns, crs=spread_speed_list[0].crs)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    # Add a static color bar outside the update function
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=LogNorm(vmin=concat_speed['speed'].min(), vmax=concat_speed['speed'].max()))
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('Speed (log scale)')

    def update(i):
        nonlocal cbar
        # Accumulate the geometries
        time_slice = spread_speed_list[i]
        time_slice_plt = fire_event_computed.iloc[i:i+1]
        nonlocal accumulated_geometries
        accumulated_geometries = pd.concat([accumulated_geometries, time_slice], ignore_index=True)
        # print(time_slice.crs)
        # if(np.unique(time_slice['speed'])>1):
        # Clear the plot and redraw
        ax.clear()
        # print(time_slice)
        accumulated_geometries.plot(column='speed', ax=ax, edgecolor='none', alpha=1.0, legend=False, cmap='coolwarm', markersize=0.9, norm=LogNorm())
            # # Add a new color bar
            # sm = plt.cm.ScalarMappable(cmap='viridis', norm=LogNorm(vmin=accumulated_geometries['speed'].min(), vmax=accumulated_geometries['speed'].max()))
            # sm._A = []  # Dummy array for the ScalarMappable
            # cbar = fig.colorbar(sm, ax=ax)
            # cbar.set_label('Speed (log scale)')

        # Set the extent to the bounds of the overall geometry before adding the basemap
        ax.set_xlim(min_x - 0.05, max_x + 0.05)
        ax.set_ylim(min_y - 0.05, max_y + 0.2)

            # Add the basemap
            # cx.add_basemap(ax, crs=fire_event_computed.crs, source='https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}{r}.png?api_key=c19b18d7-14f8-45e7-b02b-529cdd22b80e')

        # Add gridlines (optional)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set title and labels
        ax.set_title(f"Fire ID {time_slice_plt['id'].values[0]}, {time_slice_plt['eco_name'].values[0]}\nDuration: {i + 1}/{time_slice_plt['event_dur'].values[0]}, Total Area: {time_slice_plt['tot_ar_km2'].values[0]:.2f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ani = FuncAnimation(fig, update, frames=len(spread_speed_list), repeat=False)
    
    # Save the animation as a GIF
    gif_path = save_path + '_spreadspeed_animation.gif'
    ani.save(gif_path, writer=PillowWriter(fps=2))  # change fps to speed up
    print("GIF saved as:", gif_path)