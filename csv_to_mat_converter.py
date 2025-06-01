import os
import scipy.io
import pandas as pd

def process_directory(directory):
    '''This function will recursively change all .mat files to .csv files in the current file or subfiles et cetera.
    This function can be still be changed / updated by : changing the way the merge is handled, changing the way file saving is handled, ...
    '''

    # we list all items in the directory
    items = os.listdir(directory)
    
    for item in items:

        full_path = os.path.join(directory, item)
        
        # if is a directory we actually need to check inside of it
        if os.path.isdir(full_path) and item!='.git':
            # thanks to recursion and the use of the full path, we can easily recurse
            process_directory(full_path)
        else:
            # for the files we will look for .mat files
            file_name, file_extension = os.path.splitext(full_path)
            
            if file_extension == '.mat':

                # Load the .mat file
                data = scipy.io.loadmat(full_path)

                if 'trajCmds' not in data or 'trajResps' not in data:
                    continue

                # get data for both commands and responses with panda
                var_traj_comm = data['trajCmds']
                var_traj_responses = data['trajResps']
                
                # build panda dataframes
                df_traj_comm = pd.DataFrame(var_traj_comm)
                df_traj_responses = pd.DataFrame(var_traj_responses)
                
                # currently we will merge dataframes (like columns side by side)
                # might change later

                df_merged = df_traj_comm.merge(df_traj_responses, left_index=True, right_index=True, how='outer')
                
                # Save the merged dataframe to a CSV file
                df_merged.to_csv(file_name+'.csv', index=False, header=False)

# we call the function to start the recursion process
process_directory('./dataset')

