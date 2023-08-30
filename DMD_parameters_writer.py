import pickle


parameters = {
    'dmd_size_x': 1920, # horizotal pixels (number of columns)
    'dmd_size_y': 1200, # vertical pixels (number of rows)
    'save_images_folder': 'C:\\Users\\User\\Documents\\SiMView Configuration\\DMD Images',
    'save_scripts_folder': 'C:\\Users\\User\\Documents\\SiMView Configuration\\DMD Scripts',
    'params_folder': 'D:\\DMD_transform_parameters'
  }

location = 'D:\\DMD_transform_parameters'
file_name = '\\DMD_parameters.pickled'

with open(location + file_name, 'wb') as out_file:
  pickle.dump(parameters,  out_file)