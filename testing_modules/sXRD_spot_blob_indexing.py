import numpy as np


'''q_dict = {}
for spot_num in range(len(spot_df)):
    q_arr = np.array([
        -np.sin(np.radians(spot_df['tth'][spot_num])) * np.sin(np.radians(spot_df['chi'][spot_num])),
        -np.sin(np.radians(spot_df['tth'][spot_num])) * np.cos(np.radians(spot_df['chi'][spot_num])),
        1 - np.cos(np.radians(spot_df['tth'][spot_num]))
    ]) / energy_2_wavelength(15)
    q_dict[str(spot_num)] = 2 * np.pi * q_arr'''

def q_vect(tth, chi, wavelength):
    # Calculate q-vector from tth and chi angles
    '''

    '''

    arr = np.array([
        -np.sin(np.radians(tth)) * np.sin(np.radians(chi)),
        -np.sin(np.radians(tth)) * np.cos(np.radians(chi)),
        1 - np.cos(np.radians(tth))
    ])

    arr = arr / np.linalg.norm(arr)

    return (4 * np.pi * np.sin(np.radians(tth / 2)) / wavelength) * arr