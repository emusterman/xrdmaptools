import numpy as np
import time as ttime

from bluesky.plans import count, list_scan



def awful_energy_rocking_curve(e_low, e_high, e_num, dwell, peakup_N=0, replicates=1, shutter=True, xrd_dets=[]):
    # Should I use e_step or e_num??
    # One follows the xanes convention, the other the flyscanning convention


    # Define some useful variables
    e_cen = (e_high - e_low) / 2
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
        logscan_detailed('BAD_ENERGY_RC')


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
    scan_md['scan']['energies'] = e_range                                   
    scan_md['scan']['start_time'] = ttime.ctime(ttime.time())

    # Live Callbacks
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

    
    # yield from list_scan(dets, energy, e_range, md=scan_md)
    yield from check_shutters(shutter, 'Open')
    yield from subs_wrapper(list_scan(dets, energy, e_range, md=scan_md),
                            {'all' : livecallbacks})
    yield from check_shutters(shutter, 'Close')
    logscan_detailed('ENERGY_RC')


def custom_energy_rocking_curve(e_low, e_high, e_num, dwell, shutter=True, xrd_dets=[]):
    # Scan to allow intermittent peakups within invoking several individual scan IDs
    raise NotImplementedError()

    # Define some useful variables
    e_cen = (e_high - e_low) / 2
    # e_num = (e_high - e_low) // e_step
    e_range = np.linspace(e_low, e_high, e_num)

    # Defining scan metadata
    scan_md = {}
    get_stock_md(scan_md)
    scan_md['scan']['type'] = 'CUSTOM_ENERGY_RC'
    scan_md['scan']['scan_input'] = [e_low, e_high, e_num, dwell]
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
    if shutter:
        yield from mov(energy, e_cen)
        yield from peakup(shutter=shutter)

    






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