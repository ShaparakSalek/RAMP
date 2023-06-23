"""
Module contains script creating setup file for Seismic Data Container component.
"""
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    start_ind = 1
    end_ind = 5
    num_cases = end_ind - start_ind + 1
    indices = np.arange(start_ind, end_ind+1)
    folders = np.array(['data_sim{:04}'.format(ind) for ind in indices])

    setup_data = {'index': indices, 'folder': folders}
    for ind1 in range(20):
        setup_data['survey{}'.format(ind1)] = np.array([
            'data_sim{:04}_t{}.bin'.format(ind2, (ind1+1)*10) for ind2 in indices])

    df = pd.DataFrame(data=setup_data)

    df.to_csv(path_or_buf=os.sep.join([
        '..', '..', 'data', 'user', 'seismic','setup_file.csv']), sep=',', index=False)
