import sys
import os
import time


## importa classes
from environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name):
   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    
    # Instantiate the environment
    env = Env(data_folder)
    
    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")
    explorer_file = os.path.join(data_folder, "explorer_config.txt")
    
    # Instantiate agents rescuer and explorer
    list_of_rescuers = []
    list_of_rescuers.append(Rescuer(env, rescuer_file, 4, 0))
    list_of_rescuers.append(Rescuer(env, rescuer_file, 4, 1))
    list_of_rescuers.append(Rescuer(env, rescuer_file, 4, 2))
    list_of_rescuers.append(Rescuer(env, rescuer_file, 4, 3))


    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp = Explorer(env, explorer_file, list_of_rescuers, 0)
    exp2 = Explorer(env, explorer_file, list_of_rescuers, 1)
    exp3 = Explorer(env, explorer_file, list_of_rescuers, 2)
    exp4 = Explorer(env, explorer_file, list_of_rescuers, 3)


    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_100x80_225vic")
    

    main(data_folder_name)
