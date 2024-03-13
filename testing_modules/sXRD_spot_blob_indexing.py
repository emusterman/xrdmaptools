import numpy as np
from tqdm import tqdm


'''q_dict = {}
for spot_num in range(len(spot_df)):
    q_arr = np.array([
        -np.sin(np.radians(spot_df['tth'][spot_num])) * np.sin(np.radians(spot_df['chi'][spot_num])),
        -np.sin(np.radians(spot_df['tth'][spot_num])) * np.cos(np.radians(spot_df['chi'][spot_num])),
        1 - np.cos(np.radians(spot_df['tth'][spot_num]))
    ]) / energy_2_wavelength(15)
    q_dict[str(spot_num)] = 2 * np.pi * q_arr'''


def get_q_vect(tth, chi, wavelength, return_kf=False, radians=False):
    # Calculate q-vector
    if not isinstance(tth, (list, tuple, np.ndarray)):
        tth = np.asarray([tth])
        chi = np.asarray([chi])
    if len(tth) != len(chi):
        raise ValueError("Length of tth does not match length of chi.")
    
    if not radians:
        tth = np.radians(tth)
        chi = np.radians(chi)

    # Ry = np.array([[np.cos(np.radians(tth)), 0, np.sin(np.radians(tth))],
    #            [0, 1, 0],
    #            [-np.sin(np.radians(tth)), 0, np.cos(np.radians(tth))]])

    # Rz = np.array([[np.cos(np.radians(chi)), -np.sin(np.radians(chi)), 0],
    #            [np.sin(np.radians(chi)), np.cos(np.radians(chi)), 0],
    #            [0, 0, 1]])

    #ki_unit = np.broadcast_to(np.array([0, 0, 1]).reshape(3, 1, 1), (3, *tth.shape))
    ki_unit = np.broadcast_to(np.array([0, 0, 1]).reshape(3, *([1,] * len(tth.shape))),
                              (3, *tth.shape))

    # kf_unit = Rz @ Ry @ ki_unit
    # negative chi. z gets reveresed somewhere
    kf_unit = np.array([-np.sin(tth) * np.cos(-chi),
                        -np.sin(tth) * np.sin(-chi),
                        np.cos(tth)])
    
    if return_kf:
        return 2 * np.pi / wavelength * kf_unit
    
    delta_k = kf_unit - ki_unit

    # Scattering vector with origin set at transmission
    q = 2 * np.pi / wavelength * delta_k

    return q



'''def q_vect(tth, chi, wavelength):
    # Calculate q-vector

    if not isinstance(tth, (list, tuple, np.ndarray)):
        tth = np.asarray([tth])
        chi = np.asarray([chi])
    if len(tth) != len(chi):
        raise ValueError("Length of tth does not match length of chi.")
    
    kf_unit = np.array([np.sin(np.radians(tth)) * np.cos(np.radians(chi)),
                        np.sin(np.radians(tth)) * np.sin(np.radians(chi)),
                        np.cos(np.radians(tth)) - 1])

    # Scattering vector with origin set at transmission
    q = 2 * np.pi / wavelength * kf_unit

    return q'''


'''def q_vect(tth, chi, wavelength):
    # Calculate q-vector

    if not isinstance(tth, (list, tuple, np.ndarray)):
        tth = np.asarray([tth])
        chi = np.asarray([chi])
    if len(tth) != len(chi):
        raise ValueError("Length of tth does not match length of chi.")

    #Ry = np.array([[np.cos(np.radians(tth)), 0, np.sin(np.radians(tth))],
    #            [0, 1, 0],
    #            [-np.sin(np.radians(tth)), 0, np.cos(np.radians(tth))]])

    #Rz = np.array([[np.cos(np.radians(chi)), -np.sin(np.radians(chi)), 0],
    #            [np.sin(np.radians(chi)), np.cos(np.radians(chi)), 0],
    #            [0, 0, 1]])
    
    ki_unit = np.asarray([[0, 0, 1],] * len(tth)).T

    #kf_unit = ki_unit @ Ry @ Rz
    kf_unit = np.array([-np.cos(np.radians(chi)) * np.sin(np.radians(tth)),
                        np.sin(np.radians(tth)) * np.sin(np.radians(chi)),
                        np.cos(np.radians(tth))])

    # Scattering vector with origin set at transmission
    q = 2 * np.pi / wavelength * (kf_unit - ki_unit)

    return q'''



'''def old_q_vect(tth, chi, wavelength):
    # Calculate q-vector from tth and chi angles

    arr = np.array([
        -np.sin(np.radians(tth)) * np.sin(np.radians(chi)),
        -np.sin(np.radians(tth)) * np.cos(np.radians(chi)),
        1 - np.cos(np.radians(tth))
    ])

    arr = arr / np.linalg.norm(arr, axis=0)

    return (4 * np.pi * np.sin(np.radians(tth / 2)) / wavelength) * arr'''


def _initial_spot_analysis(xrdmap, SpotModel=None):
    # TODO: rewrite with spots dataframe and wavelength as inputs...

    # Extract fit stats
    print('Extracting more information from peak parameters...')
    if SpotModel is not None and any([x[:3] == 'fit' for x in xrdmap.spots.loc[0].keys()]):
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:3] == 'fit'][:6]
        prefix='fit'
    elif SpotModel is None or SpotModel == 'guess':
        interested_params = [x for x in xrdmap.spots.iloc[0].keys()
                             if x[:5] == 'guess']
        prefix='guess'

    for i in tqdm(xrdmap.spots.index):
        spot = xrdmap.spots.loc[i]

        if prefix == 'fit':
            fit_params = spot[interested_params]

            fwhm = SpotModel.get_2d_fwhm(*fit_params)
            volume = SpotModel.get_volume(*fit_params)

        elif prefix == 'guess':
            guess_params = spot[['guess_height',
                                 'guess_cen_tth',
                                 'guess_cen_chi',
                                 'guess_fwhm_tth',
                                 'guess_fwhm_chi']].values
            fwhm = GaussianFunctions.get_2d_fwhm(*guess_params, 0) # zero for theta
            volume = spot['guess_int']

        more_params = [volume, *fwhm]

        labels = ['integrated',
                  'fwhm_a',
                  'fwhm_b',
                  'rot_fwhm_tth',
                  'rot_fwhm_chi']
        labels = [f'{prefix}_{label}' for label in labels]
        for ind, label in enumerate(labels):
            xrdmap.spots.loc[i, label] = more_params[ind]
    print('done!')
    
    
    # Find q-space coordinates
    print('Converting peaks positions to q-space...', end='', flush=True)
    if prefix == 'fit':
        spot_tth = xrdmap.spots['fit_tth0'].values
        spot_chi = xrdmap.spots['fit_chi0'].values

    elif prefix == 'guess':
        spot_tth = xrdmap.spots['guess_cen_tth'].values
        spot_chi = xrdmap.spots['guess_cen_chi'].values
    
    q_values = get_q_vect(spot_tth, spot_chi, xrdmap.wavelength)

    for key, value in zip(['qx', 'qy', 'qz'], q_values):
        xrdmap.spots[key] = value
    print('done!')