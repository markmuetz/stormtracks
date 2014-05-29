import netCDF4 as nc

def process_data(settings):
    data_dir = 'data/20C_2005_Wilma'
    processed_data_dir = 'processed_data'
    ds_map = ({'fn': 'air.2005.nc',        'var_in':'air',   'var_out':'temp', 'is_level':True},
              {'fn': 'uwnd.2005.nc',       'var_in':'uwnd',  'var_out':'u',    'is_level':True},
              {'fn': 'vwnd.2005.nc',       'var_in':'vwnd',  'var_out':'v',    'is_level':True},
              {'fn': 'prmsl.2005.nc',      'var_in':'prmsl', 'var_out':'psl',  'is_level':False},
              {'fn': 'uwnd.2005.nc',       'var_in':'uwnd',  'var_out':'u10',  'is_level':True},
              {'fn': 'air.sig995.2005.nc', 'var_in':'air',   'var_out':'tsu',  'is_level':False},
	      )

    datasets = {}
    ds_out = nc.Dataset('%s/%s'%(processed_data_dir, 'c20_200508.nc'), 'w')
    try:
	for map_item in ds_map:
	    ds = nc.Dataset('%s/%s'%(data_dir, map_item['fn']))
	    datasets[map_item['fn']] = ds

	air = datasets['air.2005.nc']
	for dim in air.dimensions:
	    if dim == 'time':
		#ds_out.createDimension(dim, len(air.dimensions[dim]) / 2)
		ds_out.createDimension(dim, 60)
	    elif dim == 'level':
		ds_out.createDimension('lev', 4)
	    else:
		ds_out.createDimension(dim, len(air.dimensions[dim]))

	level_indices = []
	for i, level in enumerate(air.variables['level'][:]):
	    if level in [850, 700, 500, 300]:
		level_indices.append(i)

	print(level_indices)

	for variable in ['lat', 'lon', 'time', 'level']:
	    if variable == 'time':
		ds_out.createVariable(variable, 'f4', (variable,))
		#ds_out.variables[variable][:] = air.variables[variable][::2]
                ds_out.variables[variable][:] = air.variables[variable][888:1008:2]
	    elif variable == 'level':
		ds_out.createVariable('lev', 'f4', ('lev',))
		for i, level_index in enumerate(level_indices):
		    ds_out.variables['lev'][i] = air.variables[variable][level_index]
		print( ds_out.variables['lev'][:])
            elif variable == 'lat':
		ds_out.createVariable(variable, 'f4', (variable,))
                ds_out.variables[variable][:] = air.variables[variable][::-1]
	    else:
		ds_out.createVariable(variable, 'f4', (variable,))
		ds_out.variables[variable][:] = air.variables[variable]
	
	print(level_indices)

	for map_item in ds_map:
	    print('copy vars for %s'%map_item['fn'])
	    copy_variables(datasets[map_item['fn']], ds_out, map_item, level_indices)

    except Exception, e:
	print('problem!')
	print(e.message)
	ds_out.close()

    return datasets, ds_out

def copy_variables(ds_in, ds_out, map_item, level_indices):
    print('copying var %s to %s'%(map_item['var_in'], map_item['var_out']))
    print('in shape %s'%str(ds_in.variables[map_item['var_in']].shape))

    if map_item['var_out'] == 'u10':
	ds_out.createVariable(map_item['var_out'], 'f4', ('time', 'lat', 'lon' ))
	print('out shape %s'%str(ds_out.variables[map_item['var_out']].shape))
	#ds_out.variables[map_item['var_out']][:, :, :] = ds_in.variables[map_item['var_in']][::2, 0]
        ds_out.variables[map_item['var_out']][:, :, :] = ds_in.variables[map_item['var_in']][888:1008:2, 0, ::-1, :]
    elif map_item['is_level']:
	ds_out.createVariable(map_item['var_out'], 'f4', ('time', 'lev', 'lat', 'lon' ))
	print('out shape %s'%str(ds_out.variables[map_item['var_out']].shape))
	for i, level_index in enumerate(level_indices):
	    print('Level %i'%level_index)
	    #ds_out.variables[map_item['var_out']][:, i, :, :] = ds_in.variables[map_item['var_in']][::2, level_index]
            ds_out.variables[map_item['var_out']][:, i, :, :] = ds_in.variables[map_item['var_in']][888:1008:2, level_index, ::-1, :]
    else:
	ds_out.createVariable(map_item['var_out'], 'f4', ('time', 'lat', 'lon' ))
	print('out shape %s'%str(ds_out.variables[map_item['var_out']].shape))
	#ds_out.variables[map_item['var_out']][:, :, :] = ds_in.variables[map_item['var_in']][::2]
        ds_out.variables[map_item['var_out']][:, :, :] = ds_in.variables[map_item['var_in']][888:1008:2, ::-1, :]
    print('done')
