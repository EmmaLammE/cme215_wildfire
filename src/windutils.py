'''
This file is created by Emma Liu (liuwj@stanford.edu)
for CME 215 class project.

This codes contains utility functions that process
now-23 wind and temperature data

Examples to use these functions can be found in:
wildfire_dataprocessing.ipynb
'''

import h5py
import dask.array as da
import dask
import s3fs
import numpy as np
import xarray as xr
from dateutil import parser
from datetime import datetime
from scipy.interpolate import griddata

# Function to list all datasets and their metadata in the HDF5 file
def list_datasets_metadata(s3_path):
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(s3_path, 'rb') as s3file:
        with h5py.File(s3file, 'r') as f:
            def print_metadata(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    for attr in obj.attrs:
                        print(f"  {attr}: {obj.attrs[attr]}")
                    print("\n")
            
            f.visititems(print_metadata)

# Function to read and print the metadata from the 'meta' dataset
def print_meta_dataset(s3_path):
    with s3.open(s3_path, 'rb') as s3file:
        with h5py.File(s3file, 'r') as f:
            if 'meta' in f:
                meta_dataset = f['meta']
                print("Meta dataset attributes:")
                for attr in meta_dataset.attrs:
                    print(f"  {attr}: {meta_dataset.attrs[attr]}")
                
                print("\nMeta dataset shape:", meta_dataset.shape)
                print("Meta dataset dtype:", meta_dataset.dtype)
                
                print("\nMeta dataset contents (first 20 rows):")
                print(meta_dataset[:20])  # Print the first 20 rows of the meta dataset
                
                print("\nMeta dataset contents (non-offshore rows):")
                non_offshore_data = meta_dataset[meta_dataset['offshore'] == 0]
                print(non_offshore_data[:20])  # Print the first 20 non-offshore rows
            else:
                print("Meta dataset not found in the file.")

# # Function to read the meta dataset using Dask
def load_meta_dataset(file):
    meta_dataset = file['meta']
    meta_data = {
        'latitude': da.from_array(meta_dataset['latitude'], chunks='auto'),
        'longitude': da.from_array(meta_dataset['longitude'], chunks='auto'),
        'country': da.from_array(meta_dataset['country'], chunks='auto'),
        'state': da.from_array(meta_dataset['state'], chunks='auto'),
        'county': da.from_array(meta_dataset['county'], chunks='auto'),
        'timezone': da.from_array(meta_dataset['timezone'], chunks='auto'),
        'elevation': da.from_array(meta_dataset['elevation'], chunks='auto'),
        'offshore': da.from_array(meta_dataset['offshore'], chunks='auto'),
    }
    return meta_data

# Function to load the time index dataset using Dask
def load_time_index(file):
    time_index = da.from_array(file['time_index'], chunks='auto')
    # time_index = time_index.astype('datetime64[s]')
    return time_index

# Function to filter the meta dataset based on latitude and longitude
def filter_meta_data(meta_data, lat_min,lat_max,lon_min,lon_max):
    filtered_indices = da.where(
        (meta_data['latitude'] >= lat_min) & (meta_data['latitude'] <= lat_max) &
        (meta_data['longitude'] >= lon_min) & (meta_data['longitude'] <= lon_max)
    )[0]

    return filtered_indices

# Function to filter the time index dataset based on the specified time range
def filter_time_index(time_index, start_time, end_time):
    # Convert the string timestamps to datetime format
    datetime_array = time_index.map_blocks(lambda x: np.array([parser.parse(t.decode('utf-8')) for t in x]), dtype=object)
    # Convert the datetime objects to timestamps
    timestamp_array = datetime_array.map_blocks(lambda x: np.array([t.timestamp() for t in x]), dtype=float)
    # Convert the string timestamps to datetime format
    filtered_time_indices = da.where((timestamp_array >= start_time) & (timestamp_array <= end_time))[0]
    return filtered_time_indices

# Function to filter specific datasets based on the filtered indices
def filter_datasets_by_indices(f, spatial_indices, time_indices, dataset_names):
    filtered_data = {}

    for dataset_name in dataset_names:
        if dataset_name in f:
            dataset = f[dataset_name]
            
            # Use Dask to handle potentially large datasets
            dask_array = da.from_array(dataset, chunks='auto')
            print(f"Original shape of {dataset_name}: {dask_array.shape}")

            # Ensure indices are integers and use them for indexing
            filtered_spatial_indices = spatial_indices.astype(int)
            filtered_time_indices = time_indices.astype(int)

            # Apply both spatial and temporal filtering
            filtered_dask_array = dask_array[filtered_time_indices, :][:, filtered_spatial_indices]
            
            filtered_data[dataset_name] = filtered_dask_array
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")

    return filtered_data

# Function to convert datasets to Xarray
def convert_to_xarray(filtered_data, meta_data, filtered_indices):
    filtered_meta = {key: value[filtered_indices].compute() for key, value in meta_data.items()}
    coords = {
        'latitude': ('location', filtered_meta['latitude']),
        'longitude': ('location', filtered_meta['longitude']),
    }
    
    # Check dimensions of the filtered data
    for name, data in filtered_data.items():
        print(f"Filtered data for {name} has shape {data.shape}")
    # Compute the filtered data before creating the xarray dataset
    computed_data = {name: data.compute() for name, data in filtered_data.items()}

    data_vars = {name: (('time', 'location'), data) for name, data in computed_data.items()}

    return xr.Dataset(data_vars, coords=coords)  

def extract_grid_data(xarray_dataset, dataset_names,latitude,longitude,grid_lon, grid_lat, time_step):
    windspeed = xarray_dataset[dataset_names[0]].data[time_step, :]
    winddirection = xarray_dataset[dataset_names[1]].data[time_step, :]
    temperature = xarray_dataset[dataset_names[2]].data[time_step, :]
    # humidity = xarray_dataset[dataset_names[3]].data[time_step, :]
    # Interpolate the temperature data onto the grid
    grid_temperature = griddata((longitude, latitude), temperature/100, 
            (grid_lon, grid_lat), method='cubic')
    # grid_humidity = griddata((longitude, latitude), humidity/100, (grid_lon, grid_lat), method='cubic')
    grid_windspeed = griddata((longitude, latitude), windspeed/100, 
            (grid_lon, grid_lat), method='cubic')
    # Check for nan values and apply nearest neighbor interpolation only where nan values are present
    mask_nan = np.isnan(grid_temperature)
    grid_temperature[mask_nan] = griddata(
        (longitude, latitude), temperature / 100,
        (grid_lon[mask_nan], grid_lat[mask_nan]), method='nearest'
    )
    mask_nan = np.isnan(grid_windspeed)
    grid_windspeed[mask_nan] = griddata(
        (longitude, latitude), windspeed / 100,
        (grid_lon[mask_nan], grid_lat[mask_nan]), method='nearest'
    )
    # Convert wind direction from degrees to radians
    wind_dir_radians = np.radians(winddirection)
    
    # Calculate the U and V components of the wind
    U = -windspeed * np.sin(wind_dir_radians)
    V = -windspeed * np.cos(wind_dir_radians)

    return U,V,grid_windspeed, grid_temperature
