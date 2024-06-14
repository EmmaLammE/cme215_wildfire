'''
This file is created by Emma Liu (liuwj@stanford.edu)
for CME 251 class project.

This codes read in wildfire spread data and use SINDy to 
find the rate of spread equation.

The SINDy (pysindy package) is modified by Emma Liu. A 
modified version can be found in ./../pysindy
'''

import sys
my_path = "./../pysindy"
# Append the directory to sys.path
if my_path not in sys.path:
    sys.path.append(my_path)
import numpy as np
import pysindy as ps
from dask_geopandas import read_file
from memory_profiler import profile

def read_fire_data(fire_id, my_path):
    #### read in all data
    print('--------------------------------------------------------------\nReading in data for fire ID: ',fire_id)
    
    speed_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_spread_daily.npz'
    spread_speed = np.load(speed_path)
    spread_daily = spread_speed['arr_0'].transpose(1, 2, 0)
    spread_daily[np.isnan(spread_daily)] = 0
    # spread_daily /= spread_daily.max()/2
    print("spread_daily shape: ",spread_daily.shape)
    print("    some data points: ", spread_daily[10,:4,1],spread_daily.max())

    wind_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_wind_daily.npz'
    wind_speed = np.load(wind_path)
    # remove the first time step as initial condition
    # spread speed only comes after 2nd step
    wind_daily = wind_speed['arr_0'].transpose(1, 2, 0)
    wind_daily[np.isnan(wind_daily)] = 0
    wind_daily /= wind_daily.max()
    print("wind_daily shape  : ",wind_daily.shape)
    print("    some data points: ", wind_daily[10,:4,1])

    temp_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_temp_daily.npz'
    temp_speed = np.load(temp_path)
    temp_daily = temp_speed['arr_0'].transpose(1, 2, 0)
    temp_daily[np.isnan(temp_daily)] = 0
    temp_daily /= temp_daily.max()
    print("temp_daily shape  : ",temp_daily.shape)
    print("    some data points: ", temp_daily[10,:4,1])

    topo_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_topo.npz'
    topo = np.load(topo_path)
    topo = topo['arr_0']
    topo = np.tile(topo[:, :, np.newaxis], (1, 1, wind_daily.shape[2]))
    topo /= topo.max()
    print("topo shape        : ",topo.shape)
    print("    some data points: ", topo[10,:4,1])
    
    slope_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_slope.npz'
    slope = np.load(slope_path)
    slope = slope['arr_0']
    slope = np.tile(slope[:, :, np.newaxis], (1, 1, wind_daily.shape[2]))
    slope /= slope.max()
    print("slope shape       : ",slope.shape)
    print("    some data points: ", slope[10,:4,1])

    NDVI_path = my_path + 'derived_data/fire_ID_'+str(fire_id)+'_NDVI.npz'
    NDVI = np.load(NDVI_path)
    NDVI = NDVI['arr_0']
    NDVI = np.tile(NDVI[:, :, np.newaxis], (1, 1, wind_daily.shape[2]))
    NDVI[NDVI<=0]=0
    print("NDVI shape        : ",NDVI.shape)
    print("    some data points: ", NDVI[10,:4,1])
    # print("##############################################################\n")
    return spread_daily, wind_daily, temp_daily, topo, slope, NDVI

def tidy_input_data(spread_daily, wind_daily, temp_daily, topo, slope, NDVI):
    ###### combine all data ########
    grid_size = wind_daily.shape[0]
    t = np.linspace(0,1,spread_daily.shape[2]) #np.linspace(0,spread_daily.shape[2]-1,spread_daily.shape[2])*3600*24
    dt = (t[1]-t[0])
    # wind_daily.shape[2] = 21
    s = np.zeros((grid_size, grid_size, len(t), 1))
    control_params = np.zeros((grid_size, grid_size, len(t), 4))
    s[:,:,:,0] = spread_daily
    control_params[:, :, :, 0] = wind_daily
    control_params[:, :, :, 1] = temp_daily
    # control_params[:, :, :, 2] = topo
    control_params[:, :, :, 2] = slope
    control_params[:, :, :, 3] = NDVI
    # take difference along 3rd dimension, i.e. take time derivative
    s_dot = ps.FiniteDifference(axis=2)._differentiate(s, dt)
    print("--------------------------------------------------------------\nInput data")
    print("unknown var s (area burned) shape                   : ", s.shape)
    print("control parameters [wind,temp,topo,slope,NDVI] shape: ", control_params.shape)
    print("dt                                                  : ", dt)
    print("grid size                                           : ", grid_size,",",grid_size)
    return s, s_dot, t, dt, grid_size, control_params

def make_grid(fire_id,my_path,grid_size):
    ##### make grid ######
    # !!!! currently sherlock cannot do .compute()
    # !!!! probably due to package version compatability issues
    # !!!! so using hard coded bounding box for now
    # fire_event_path = my_path + 'derived_data/fireID_'+str(fire_id)+'_fire_event_computed.gpkg'
    # gdf = read_file(fire_event_path, npartitions=10)
    # concat_speed = gdf.compute()#.to_crs(epsg=4326)

    # min_x = concat_speed.bounds.minx.min()
    # max_x = concat_speed.bounds.maxx.max()
    # min_y = concat_speed.bounds.miny.min()
    # max_y = concat_speed.bounds.maxy.max()
    
    # # grid data
    if (fire_id == 429894):
        min_x, max_x = -10559129.466021484,-10425230.090944916
        min_y, max_y = 4385485.518290994,4488806.25407672
    elif(fire_id == 394701):
        min_x, max_x = -10421525.589212691,-10362682.874213647
        min_y, max_y = 4407261.215967806,4433208.728093371
    elif(fire_id == 387281):
        min_x, max_x = -10191259.169098318,-10140292.770280248
        min_y, max_y = 4696368.351081226,4734825.306553044
    elif(fire_id == 429837):
        min_x, max_x = -10767620.188459046,-10697194.655546803
        min_y, max_y = 4126493.7097518896,4181166.6103021842
    elif(fire_id == 430044):
        min_x, max_x = -10655035.198342765,-10606848.67582386
        min_y, max_y = 4011592.156052966,4045879.2970760325
    elif(fire_id == 387060):
        min_x, max_x = -10567005.782202458,-10530865.39031328
        min_y, max_y = 4170508.4178220415,4199235.806246772
    elif(fire_id == 430137):
        min_x, max_x = -10363148.186930176,-10314498.351694744
        min_y, max_y = 4559227.786988962,4583785.360964943
    elif(fire_id == 368196):
        min_x, max_x = -10277435.334372511,-10241294.942483332
        min_y, max_y = 4629188.007184679,4652355.6430110745
    elif(fire_id == 387635):
        min_x, max_x = -10374267.692126846,-10293185.96673446
        min_y, max_y = 4502703.635572556,4551816.783524517
    elif(fire_id == 366743):
        min_x, max_x = -10242686.880632916,-10177357.78760248
        min_y, max_y = 4680615.718719277,4714902.859742343
    elif(fire_id == 430026):
        min_x, max_x = -10410869.396732552,-10319594.791576551
        min_y, max_y = 4395678.398054608,4436451.917109066
    else:
        print("!!!!!!!!!!\nDynamically lazy computing somehow does not work on Sherlock due to package version issue.\nBounding box data for fire ID:",fire_id, " not currently coded.")
        print("go to google drive wildfire_dataprep.ipynb landsat (NDVI) section to look for data. Exiting program.")
        sys.exit()
    # shift and scale the x y domain so that they are centered around 0
    mean_x,mean_y = (min_x+max_x)/2,(min_y+max_y)/2
    min_x -= mean_x
    max_x -= mean_x
    min_y -= mean_y
    max_y -= mean_y
    max_dy = np.min(np.abs([min_x,max_x,min_y,max_y]))
    min_x,max_x,min_y,max_y = min_x/max_dy,max_x/max_dy,min_y/max_dy,max_y/max_dy
    # min_x, max_x = -1,1
    # min_y, max_y = -1,1
    x_uniform = np.linspace(min_x, max_x, grid_size + 1)
    y_uniform = np.linspace(min_y, max_y, grid_size + 1)
    x = x_uniform[:grid_size]
    y = y_uniform[:grid_size]
    dx,dy = x[2]-x[1], y[2]-y[1]
    print("a lim, y lim                              : ",x.min(),x.max(),", ",y.min(),y.max())
    print("dx, dy                                    : ",x[2]-x[1],", ",y[2]-y[1])
    # print("##############################################################")
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy

def train(ps,pde_lib,s_train,t,control_params):
    ######## train ##########
    print("--------------------------------------------------------------\nStart Training...")
    print('STLSQ model: ')
    optimizer = ps.STLSQ(threshold=50, alpha=1e-5, 
                        normalize_columns=True, max_iter=200)
    ensemble_optimizer = ps.EnsembleOptimizer(
        optimizer,
        bagging=True,
        n_models=10,
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer,
                    feature_names=["s", "u", "T", "a", "V"])
    model.fit(s_train,t,u=control_params)
    model.print()
    coeffs_shape = model.coefficients()[0].shape
    # print(model.coefficients())
    return model, coeffs_shape





if __name__ == "__main__":
    
    my_path = "/home/groups/jsuckale/liuwj/cme215/"
    fire_ids = [387281,429837,430044,394701,429837
                ] #,429837,430044,394701,387060,430137,368196,387635,366743,430026
    print("########################################################################\n########################################################################")
    print("Total num of fire events to simulate: ", len(fire_ids))
    s_train, s_dot_train, control_params_train, t_train = [],[],[],[]
    for fire_id in fire_ids:
        print("\nStart fire id ",fire_id,"...")
        
        spread_daily, wind_daily, temp_daily, topo, slope, NDVI = read_fire_data(fire_id,my_path)

        # s - spread [area], s_dot - time derivative of s [m^2/s]
        # control params - [u,z,alpha,ND]
        s, s_dot, t, dt, grid_size, control_params = tidy_input_data(spread_daily, wind_daily, temp_daily, topo, slope, NDVI)
        X, Y, dx, dy = make_grid(fire_id,my_path,grid_size)
        
        ###### split train and test data ######
        spatial_grid = np.asarray([X, Y]).T
        # s_train = s # s_train is np array type
        # s_dot_train = s_dot

        s_train.append(s)
        s_dot_train.append(s_dot)
        control_params_train.append(control_params)
        t_train.append(t)

    # print(type(s_train))
    # a = np.array(s_train)
    # print(a.shape)
        ###### define pde function lib #####
    pde_lib = ps.PDELibrary(
                    function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
                    derivative_order=2,
                    spatial_grid=spatial_grid,
                    include_bias=True,
                    is_uniform=True,
                    periodic=True
    )

    model, coeffs_shape = train(ps,pde_lib,s_train,t_train,control_params_train)
    print("coeffs_shape: ",coeffs_shape)
    print("Finished fire id ",fire_id)
    print("########################################################################")


    # ######## test data ##########
    fire_id_test = 429894
    spread_daily, wind_daily, temp_daily, topo, slope, NDVI = read_fire_data(fire_id_test,my_path)
    s, s_dot, t_test, dt, grid_size, control_params_test = tidy_input_data(spread_daily, wind_daily, temp_daily, topo, slope, NDVI)
    s_test = s
    s_dot_test = s_dot
    

    u_dot_stlsq = model.predict(s_test,u=control_params_test)
    np.save("prediction.npz",u_dot_stlsq)
    print("R2 score: ", model.score(s_test, u=control_params_test, t=t_test))