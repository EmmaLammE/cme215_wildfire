'''
This file is created by Emma Liu (liuwj@stanford.edu)
for CME 215 class project.

This codes contains utility functions that process
SRTM data to get topography and slope

Examples to use these functions can be found in:
wildfire_dataprocessing.ipynb
'''

from pyproj import CRS, Transformer
import tifffile
import numpy as np
import sys

def get_WGS84_latlon_grid(lat_min, lat_max,lon_min, lon_max,Nx,Ny):
    crs_wgs84 = CRS.from_string('EPSG:4326')  # EPSG:4326 represents WGS84

    # Create a transformer to convert between WGS84 and the local coordinate system
    transformer = Transformer.from_crs(crs_wgs84, crs_wgs84, always_xy=True)

    # Generate the latitude and longitude grids
    latitude = np.linspace(lat_min, lat_max, Ny)
    longitude = np.linspace(lon_min, lon_max, Nx)
    grid_lon, grid_lat = np.meshgrid(longitude, latitude)

    # Convert the grid points to WGS84 coordinates
    lon_flat, lat_flat = grid_lon.flatten(), grid_lat.flatten()
    lon_wgs84, lat_wgs84 = transformer.transform(lon_flat, lat_flat)

    # Reshape the WGS84 coordinates back into a grid
    lon = lon_wgs84.reshape(grid_lon.shape)
    lat = lat_wgs84.reshape(grid_lat.shape)

    return lat, lon

def calculate_aspect(dem, cellsize):
    """Calculate the aspect from the DEM data."""
    # Compute gradients
    dzdx = np.gradient(dem, axis=1) / cellsize
    dzdy = np.gradient(dem, axis=0) / cellsize

    # Compute aspect in radians and convert to degrees
    aspect = np.arctan2(-dzdx, dzdy)  # Negate dzdx to align with direction standards
    aspect_deg = np.degrees(aspect)

    # Convert to compass direction
    aspect_deg = np.where(aspect_deg < 0, 90 - aspect_deg, 450 - aspect_deg) % 360

    return aspect_deg

def get_topo_data(fire_id,my_path):
    if(fire_id==357266):
        # fire id = 357266
        topo_path_1 = my_path + 'data/srtm_357266_1.tif' # 37, 36, -100, -101
        topo_path_2 = my_path + 'data/srtm_357266_2.tif' # 37, 36,  -99, -100
        topo_path_3 = my_path + 'data/srtm_357266_3.tif' # 38, 37, -100, -101
        topo_path_4 = my_path + 'data/srtm_357266_4.tif' # 38, 37,  -99, -100
        lat_max, lat_min = 38, 36
        lon_max, lon_min = -99,-101

        topo1 = tifffile.imread(topo_path_1)
        topo2 = tifffile.imread(topo_path_2)
        topo3 = tifffile.imread(topo_path_3)
        topo4 = tifffile.imread(topo_path_4)

        topo_12 =np.hstack([topo1,topo2]); 
        topo_34 =np.hstack([topo3,topo4]); 

        topo_all = np.vstack([topo_34,topo_12])

        topo_all = np.flipud(topo_all)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        
    elif(fire_id==429837):
        # fire_id==429837
        topo_path = my_path + 'data/srtm_429837_1.tif' # 38, 37, -121, -122
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 38, 37
        lon_max, lon_min = -121,-122
    elif(fire_id==429894):
        # fire id = 429894
        topo_path_1 = my_path + 'data/srtm_429894_1.tif' # 37, 36, -100, -101
        topo_path_2 = my_path + 'data/srtm_429894_2.tif' # 37, 36,  -99, -100
        topo_path_3 = my_path + 'data/srtm_429894_3.tif' # 38, 37, -100, -101
        topo_path_4 = my_path + 'data/srtm_429894_4.tif' # 38, 37,  -99, -100
        lat_max, lat_min = 41, 38
        lon_max, lon_min = -122,-124

        topo1 = tifffile.imread(topo_path_1)
        topo2 = tifffile.imread(topo_path_2)
        topo3 = tifffile.imread(topo_path_3)
        topo4 = tifffile.imread(topo_path_4)

        topo_12 =np.hstack([topo2,topo1]); 
        topo_34 =np.hstack([topo4,topo3]); 

        topo_all = np.vstack([topo_34,topo_12])

        topo_all = (topo_all)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
    elif(fire_id==387635):
        # fire_id==387635
        topo_path = my_path + 'data/srtm_387635.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 41, 40
        lon_max, lon_min = -122,-123
    elif(fire_id==376051):
        # fire_id==376051
        topo_path1 = my_path + 'data/srtm_376051_1.tif' # 35, 34, -119,-120
        topo_path2 = my_path + 'data/srtm_376051_2.tif' # 35, 34, -118,-119
        topo1 = tifffile.imread(topo_path1)
        topo2 = tifffile.imread(topo_path2)
        topo_all =np.hstack([topo1,topo2]); 
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 35, 34
        lon_max, lon_min = -118,-120
    elif(fire_id==430026):
        # fire_id==430026
        topo_path1 = my_path + 'data/srtm_430026_1.tif' # 40, 39, -121,-122
        topo_path2 = my_path + 'data/srtm_430026_2.tif' # 40, 39, -120,-121
        topo1 = tifffile.imread(topo_path1)
        topo2 = tifffile.imread(topo_path2)
        topo_all =np.hstack([topo1,topo2]); 
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 40, 39
        lon_max, lon_min = -120,-122
    elif(fire_id==394701):
        # fire_id==394701
        topo_path = my_path + 'data/srtm_394701.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 40, 39
        lon_max, lon_min = -121,-122
    elif(fire_id==387281):
        # fire_id==3387281
        topo_path1 = my_path + 'data/srtm_387281_1.tif' # 42, 43, -123, -124
        topo_path2 = my_path + 'data/srtm_387281_2.tif' # 42, 43, -124, -125
        topo1 = tifffile.imread(topo_path1)
        topo2 = tifffile.imread(topo_path2)
        topo_all =np.hstack([topo2,topo1]); 
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 43, 42
        lon_max, lon_min = -123,-125
    elif(fire_id==430044):
        # fire_id==430044
        topo_path = my_path + 'data/srtm_430044.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 37, 36
        lon_max, lon_min = -118,-119
    elif(fire_id==387060):
        topo_path = my_path + 'data/srtm_430044.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 38, 37
        lon_max, lon_min = -119,-120
    elif(fire_id==430137):
        topo_path = my_path + 'data/srtm_430137.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 42, 41
        lon_max, lon_min = -123,-124
    elif(fire_id==368196):
        topo_path = my_path + 'data/srtm_368196.tif' # 41, 40, -122, -123
        topo_all = tifffile.imread(topo_path)
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 42, 41
        lon_max, lon_min = -123,-124
    elif(fire_id==366743):
        topo_path1 = my_path + 'data/srtm_366743_1.tif' # 42, 43, -123, -124
        topo_path2 = my_path + 'data/srtm_366743_2.tif' # 42, 43, -124, -125
        topo1 = tifffile.imread(topo_path1)
        topo2 = tifffile.imread(topo_path2)
        topo_all =np.hstack([topo1,topo2]); 
        Ny = np.size(topo_all,0)
        Nx = np.size(topo_all,1)
        lat_max, lat_min = 43, 42
        lon_max, lon_min = -123,-125
    else:
        print("Not downloaded SRTM data for fire ID:",fire_id)
        print("go to https://earthexplorer.usgs.gov/ to download")
        sys.exit()
    return np.flipud(topo_all), Nx, Ny, lat_max,lat_min,lon_max,lon_min


