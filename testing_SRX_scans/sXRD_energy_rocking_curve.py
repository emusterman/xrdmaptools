import numpy as np
import time as ttime
from itertools import product

from bluesky.plans import count, list_scan


def setup_xrd_dets(dets, dwell, N_images):
    # Convenience function for setting up xrd detectors

    dets_by_name = {d.name : d for d in dets}

    # Setup merlin
    if 'merlin' in dets_by_name:
        xrd = dets_by_name['merlin']
        # Make sure we respect whatever the exposure time is set to
        if (dwell < 0.0066392):
            print('The Merlin should not operate faster than 7 ms.')
            print('Changing the scan dwell time to 7 ms.')
            dwell = 0.007
        # According to Ken's comments in hxntools, this is a de-bounce time
        # when in external trigger mode
        xrd.cam.stage_sigs['acquire_time'] = 0.75 * dwell  # - 0.0016392
        xrd.cam.stage_sigs['acquire_period'] = 0.75 * dwell + 0.0016392
        xrd.cam.stage_sigs['num_images'] = 1
        xrd.stage_sigs['total_points'] = N_images
        xrd.hdf5.stage_sigs['num_capture'] = N_images
        del xrd

    # Setup dexela
    if 'dexela' in dets_by_name:
        xrd = dets_by_name['dexela']
        xrd.cam.stage_sigs['acquire_time'] = dwell
        xrd.cam.stage_sigs['acquire_period'] = dwell
        del xrd

def energy_rocking_curve(xrd_dets, e_low, e_high, e_num, dwell,
                         shutter=True, peakup=True):
    # Should I use e_step or e_num??
    # One follows the xanes convention, the other the flyscanning convention


    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'ENERGY_RC'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total']),
                     LivePlot('dexela_stats2_total', x='energy_energy')]

    # Define detectors
    dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, e_num)

    # Move to center energy and perform peakup
    if peakup:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)

    
    # yield from list_scan(dets, energy, e_range, md=scan_md)
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, energy, e_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')
    logscan_detailed('ENERGY_RC')


def approx_Laue_scan(xrd_dets, e_low, e_high, e_num, dwell,
                     peakup=True, shutter=True):
    # energy rocking curve with fixed u_gap position
    # Thanks to Denis for the idea

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'LAUE_SCAN'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total']),
                     LivePlot('dexela_stats2_total', x='energy_energy')]

    # Define detectors
    dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, e_num)

    # Move to center energy and perform peakup
    if peakup:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)

    energy.move_u_gap.put(False)
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, energy, e_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')
    energy.move_u_gap.put(True)
    logscan_detailed('LAUE_SCAN')


def xrd_energy_scan(xrd_dets, e_low, e_high, e_num, dwell,
                    fix_u_gap=False, peakup=True, shutter=True):
    
    if fix_u_gap:
        scan_type = 'LAUE_SCAN'
    else:
        scan_type = 'ENERGY_RC'

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = scan_type
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total']),
                     LivePlot('dexela_stats2_total', x='energy_energy')]

    # Define detectors
    dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, e_num)

    # Move to center energy and perform peakup
    if peakup:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)

    if fix_u_gap:
        energy.move_u_gap.put(False)

    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, energy, e_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')

    if fix_u_gap:
        energy.move_u_gap.put(True)
    
    logscan_detailed(scan_type)


# A static xrd measurement without changing energy or moving stages
def static_xrd(xrd_dets, num, dwell,
               peakup=False, shutter=True):
    #raise NotImplementedError()

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'STATIC_XRD'
    scan_md['scan']['scan_input'] = [num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total']),
                     LivePlot('dexela_stats2_total', x='energy_energy')]

    # Define detectors
    dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, num)

    # Move to center energy and perform peakup
    if peakup:  # Find optimal c2_fine position
        print('Performing center energy peakup.')
        yield from peakup(shutter=shutter)

    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(count(dets, num, md=scan_md), # I guess dwell info is carried by the detector
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')
    
    logscan_detailed('STATIC_XRD')


# Iterate through all relevant detector settings to collect a series of dark-field images
def acquire_dark_fields(xrd_dets, num, param_dict=None):

    if param_dict is None and 'dexela' in [d.name for d in xrd_dets]:
        param_dict = {'dwell' : [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2],
                      'binning_mode' : [0, 4, 8],
                      'full_well_mode' : [0, 1]}

    # It may just be best to force a list input
    if isinstance(param_dict, dict):
        param_dict = [param_dict] # Convert to list to match xrd_dets

    if len(xrd_dets) != len(param_dict):
        raise ValueError('Number of parameter dictionaries does not match number of XRD detectors.')
    
    scan_dict = []
    num_scans = []
    for param_dict_i in param_dict:
        if 'dwell' not in param_dict_i.keys():
            raise ValueError('Must specify dwell time in param_dict for each detector.')
        
        out = list(product(*param_dict_i.values()))
        scan_dict_i = dict(zip(param_dict_i.keys(), np.asarray(out).T))
        scan_dict.append(scan_dict_i)
        num_scans.append(len(out))
    
    # A couple more to check before performing scans
    dets = [sclr1] + xrd_dets
    yield from check_shutters(True, 'Close')

    # Iterate through all planned scans
    for scan_num in range(np.max(num_scans)):
        # Check and remove detector from scans if all combinations have been measured
        for i in range(len(xrd_dets)):
            if scan_num + 1 > num_scans[i]:
                xrd_dets.pop(i)

        # Stage detectors for each particular scan
        outstr = f'Scan Number {scan_num + 1}: Acquring dark-field for {[d.name for d in xrd_dets]}.'
        for xrd, params in zip(xrd_dets, scan_dict):
            # Only supports dexela and merlin
            if xrd.nam == 'dexela':
                xrd.cam.stage_sigs['acquire_time'] = params['dwell'][scan_num]
                xrd.cam.stage_sigs['acquire_period'] = params['dwell'][scan_num]

                for key, value in params.items():
                    if key == 'dwell':
                        continue
                    xrd.cam.stage_sigs[key] = value[scan_num]
            
            if xrd.name == 'merlin':
                # Make sure we respect whatever the exposure time is set to
                if (params['dwell'][scan_num] < 0.0066392):
                    print('The Merlin should not operate faster than 7 ms.')
                    print('Changing the scan dwell time to 7 ms.')
                    params['dwell'][scan_num] = 0.007
                # According to Ken's comments in hxntools, this is a de-bounce time
                # when in external trigger mode
                xrd.cam.stage_sigs['acquire_time'] = 0.75 * params['dwell'][scan_num]  # - 0.0016392
                xrd.cam.stage_sigs['acquire_period'] = 0.75 * params['dwell'][scan_num] + 0.0016392
                xrd.cam.stage_sigs['num_images'] = 1
                xrd.stage_sigs['total_points'] = num
                xrd.hdf5.stage_sigs['num_capture'] = num

                for key, value in params.items():
                    if key == 'dwell':
                        continue
                    xrd.cam.stage_sigs[key] = value[scan_num]
            
            # Append values to output string
            outstr += f'\n{xrd.name} parameters are:'
            for key, value in params.items():
                if key != 'dwell':
                    value_name = getattr(xrd.cam, key).get()
                else:
                    value_name = value
                outstr += f'\n\t{key} : {value[scan_num]}'

        # Defining scan metadata
        scan_md = {}
        get_stock_md(scan_md)
        scan_md['scan']['type'] = 'AREA_DARK_FIELD'
        scan_md['scan']['scan_input'] = [num, [d.keys() for d in param_dict]]
        scan_md['scan']['dwell'] = dwell
        scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
        scan_md['scan']['energy'] = f'{energy.energy.position:.5f}'                                 
        scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

        # Actually acquireing the scan!
        print(outstr)
        yield from count(dets, num, md=scan_md)
        
        logscan_detailed('AREA_DARK_FIELD')
    

    





'''def awful_energy_rocking_curve(e_low, e_high, e_num, dwell, peakup_N=0, replicates=1, shutter=True, xrd_dets=[]):
    # Should I use e_step or e_num??
    # One follows the xanes convention, the other the flyscanning convention


    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    # e_num = (e_high - e_low) // e_step
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'BAD_ENERGY_RC'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell, replicates]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energies'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Define detectors
    if xrd_dets == []:
        raise ValueError("Must define an xrd detector!")
    else:
        dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, e_num)

    # Move to center energy and perform peakup
    if peakup_N == 0 and shutter:
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)
    elif peakup_N == 0 and not shutter:
        print('Shutter set to "False". Skipping center peakup.')
    elif peakup_N != 0:
        print(f'Peakups will be performed for every {peakup_N} energies.')

    # Cycle through energies
    for i, e in enumerate(e_range):
        # Move energy
        yield from mov(energy, e)

        # Periodically perform peakup to maximize signal
        if (peakup_N != 0) and (i % peakup_N == 0):
            yield from peakup(shutter=shutter) #shutter=shutter necessary?

        
        yield from count(dets, num=replicates, md=scan_md)
        logscan_detailed('BAD_ENERGY_RC')'''


'''def bad_approx_Laue_scan(xrd_dets, e_low, e_high, e_num, dwell, harmonic=3,
                     peakup=True, shutter=True):
    # Hold undulator constant and just scan mono
    # Thanks to Denis for the idea

    # Define some useful variables
    e_cen = (e_high + e_low) / 2
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'BAD_LAUE_SCAN'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
    scan_md['scan']['dwell'] = dwell
    scan_md['scan']['detectors'] = [sclr1.name] + [d.name for d in xrd_dets]
    scan_md['scan']['energies'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
    # What does energy_energy read??
    livecallbacks = [LiveTable(['energy_energy', 'dexela_stats2_total']),
                     LivePlot('dexela_stats2_total', x='energy_energy')]

    # Define detectors
    dets = [sclr1] + xrd_dets # include xs just for fun?
    setup_xrd_dets(dets, dwell, e_num)

    # Move to center energy and perform peakup
    if peakup:
        print('Performing center energy peakup.')
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)

    # Get bragg (dcm_roll) and dcm pitch positions
    # Just energy psuedomotor without u_gap
    # there might be a flag to turn that off
    bragg_vals, c2x_vals = [], []
    for e in e_range:
        BraggRBV, C2X, ugap = energy.energy_to_positions(e, harmonic, u_detune=0) # Not sure what u_detune is???
        bragg_vals.append(BraggRBV)
        c2x_vals.append(C2X)
    
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets,
                                      energy.bragg, bragg_vals, # direct motor would be dcm.c1_roll
                                      energy.c2_x, c2x_vals, # direct motor would dcm.c2_pitch
                                      md=scan_md),
                                      {'all' : livecallbacks})

    yield from check_shutters(shutter, 'Close')
    logscan_detailed('BAD_LAUE_SCAN')'''



    