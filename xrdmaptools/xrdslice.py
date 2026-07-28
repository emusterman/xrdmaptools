import numpy as np
import os
import h5py
import pandas as pd
from tqdm import tqdm
import functools

# Local imports
from xrdmaptools.xrdbasescan import XRDBaseScan
from xrdmaptools.xrdmap import XRDMap
from xrdmaptools.xrdrockingcurve import XRDRockingCurve
from xrdmaptools.utilities.utilities import (
    pathify,
    _check_dict_key,
    copy_docstring
    )
from xrdmaptools.utilities.math import (
    wavelength_2_energy,
    energy_2_wavelength
)
from xrdmaptools.io.hdf_utils import (
    overwrite_attr,
    get_large_map_slices
    )
from xrdmaptools.io.db_io import (
    load_data,
    get_scantype
    )
from xrdmaptools.reflections.spot_blob_indexing import _initial_spot_analysis
from xrdmaptools.reflections.spot_models import GaussianFunctions
from xrdmaptools.reflections.spot_blob_search import (
    find_blobs_spots,
    find_spot_stats,
    make_stat_df,
    remake_spot_list,
    fit_spots
    )
from xrdmaptools.plot.interactive_map import (
    Interactive2DPlot,
    Interactive1DPlot,
    Interactive1DWindowSumPlot,
    Interactive1DWindowCOMPlot,
)
from xrdmaptools.plot.general import (
    return_plot_wrapper,
    plot_map,
    )


class XRDSlice(XRDBaseScan):
    """
    Class for analyzing and processing XRD data acquired every pixel
    along a 1D spatial map while rocking in energy or angle.

    Parameters
    ----------
    pos_dict : dict, optional
        Dictionary of 2D numpy arrays matching the map shape with
        position values used to inform and scale map plots.
    rocking_axis : {'energy', 'angle'}, optional
        String indicating which axis was used as the rocking axis to
        scan in reciprocal space. Will accept variations of 'energy',
        'wavelength', 'angle' and 'theta', but stored internally as
        'energy', or 'angle'. Defaults to 'energy'.
    check_init_sets : bool, optional
        Flag whether to overwrite data in the HDF file if available. By
        default this is set to False and should only be True when
        instantiating from the HDF file.
    xrdbasekwargs : keyword arguments, optional
        All other kwargs for parent XRDBaseScan class.
    """

    # Class variables
    _hdf_type = 'xrdslice'

    def __init__(self,
                 pos_dict=None,
                 rocking_axis=None,
                 check_init_sets=False,
                 **xrdbasekwargs
                 ):



        XRDBaseScan.__init__(
            self,
            map_labels=['rocking_ind',
                        'map_ind'],
            check_init_sets=check_init_sets,
            **xrdbasekwargs,
            )

        # Set position dictionary
        if 'check_init_sets' in xrdbasekwargs:
            check_init_sets = xrdbasekwargs['check_init_sets']
        self.pos_dict = None
        if pos_dict is not None:
            self.set_positions(pos_dict,
                               check_init_sets=check_init_sets)

        # Interpolate positions
        # Can happen before or after swapped_axes
        # Only if interpolated positions are not already determined
        if (self.scan_input is not None
            and len(self.scan_input) != 0
            and ('interp_x' not in self.pos_dict 
                 or 'interp_y' not in self.pos_dict)):
            self.interpolate_positions(check_init_sets=check_init_sets)

        # Check for changes. Wavelength will depend on energy
        if (np.all(~np.isnan(self.energy))
            and np.all(~np.isnan(self.theta))):
            # If all the same raise error
            if all([
                all(np.round(i, 4) == np.round(self.energy[0], 4)
                    for i in self.energy),
                all(np.round(i, 4) == np.round(self.theta[0], 4)
                    for i in self.theta)
            ]):
                err_str = ('Rocking curves must be constructed '
                           + 'by varying energy/wavelength or '
                           + 'theta. Given values are constant.')
                raise ValueError(err_str)
        
        # Find rocking axis
        self._set_rocking_axis(rocking_axis=rocking_axis)
        
        # Enable features
        # This enables theta usage, but there are no offset
        # options to re-zero stage rotation
        if (not self.use_stage_rotation
            and self.rocking_axis == 'angle'):
            self.use_stage_rotation = True

        # I am not sure if this whole block is needed?
        @XRDBaseScan._protect_hdf()
        def save_extra_attrs(self): # Not sure if this needs self...
            attrs = self.hdf[self._hdf_type].attrs
            overwrite_attr(attrs, 'energy', self.energy)
            overwrite_attr(attrs, 'wavelength', self.wavelength)
            overwrite_attr(attrs, 'theta', self.theta)
            overwrite_attr(attrs, 'rocking_axis', self.rocking_axis)
        save_extra_attrs(self)


    @copy_docstring(XRDRockingCurve.__str__)
    def __str__(self):
        return XRDRockingCurve.__str__(self)

    
    @copy_docstring(XRDRockingCurve.__repr__)
    def __repr__(self):
        return XRDRockingCurve.__repr__(self)


    ################################
    ### Loading data into XRDMap ###
    ################################
        

    @classmethod
    def from_db(cls,
                scan_id=-1,
                broker='manual',
                wd=None,
                filename=None,
                save_hdf=True,
                data_keys=None,
                xrd_dets=None,
                repair_method='fill',
                **kwargs):

        """
        Instantiate from data in database.

        Load data and metadata from database to create a new instance.
        This will copy, compile, and write the raw data into a single
        HDF by default.

        Parameters
        ----------
        scan_id : int, optional
            Scan ID of scan to load data from database. -1 by default
            which loads from the most recent scan.
        broker : {"manual", "tiled", "databroker"}, optional
            Broker/method used to load the data. "manual" will load
            data directly from raw HDF files, while "tiled" and
            "databroker" use their respective libraries. "manual" by
            default.
        wd : path str, optional
            Working directly used to write HDF file if saved. Uses
            current working directory if none provided.
        filename : str, optional
            Filename used used to write HDF file if saved. Uses
            "scan<scan_id>_xrdmap.h5" by default.
        save_hdf : bool, optional
            Flag to enable writing data to HDF file. True by default.
        data_keys : list, optional
            List of data keys to be loaded into the instance. Not all
            data keys are supported or will be used. ['enc1', 'enc2',
            'i0', 'i0_time', 'im', 'it'] be default along with any
            either the merlin or dexela area detectors if used.
        repair_method : {"fill", "flatten", "replace"}, optional
            Repair method used for missing data if data broker is
            "manual". "fill" auto-pads missing pixels with zero and is
            the default option. This also matches the behavior when
            broker is "tiled". "flatten" will flatten the entire
            dataset into a single line and is not recommended.
            "replace" will replace rows with missing data by the
            previously full row, matching the behavior of pyXRF.
        kwargs : keyword arguments, optional
            All other keyward arguments passed to __init__.
        
        Returns
        -------
        xrdslice : XRDSlice
            Instance of XRDMap class with all data and metadata from
            the designaed scan ID.

        Raises
        ------
        OSError if working directory does not exist.
        RuntimeError if scan type is incorrect for XRDSlice.
        """
        
        if wd is None:
            wd = os.getcwd()
        else:
            if not os.path.exists(wd):
                err_str = f'Cannot find directory {wd}'
                raise OSError(err_str)

        # Get scantype information from check...
        if broker == 'manual':
            temp_broker = 'tiled'
        else:
            temp_broker = broker
        scantype = get_scantype(scan_id,
                                broker=temp_broker)
        if scantype not in ['XAS_SLICE']:
            err_str = (f"Scan of type {scantype} not currently "
                       + "supported. Scan type must be 'XAS_SLICE'.")
            raise RuntimeError(err_str)
    
        # No fluorescence key
        pos_keys = ['enc1', 'enc2', 'nano_stage_x', 'nano_stage_y', 'nano_stage_topx']
        rock_keys = ['energy_energy', 'nano_stage_th']
        sclr_keys = ['i0', 'i0_time', 'im', 'it']
        
        if data_keys is None:
            data_keys = pos_keys + sclr_keys + rock_keys

        (data_dict,
         scan_md,
         data_keys,
         xrd_dets) = load_data(scan_id=scan_id,
                               broker=broker,
                               detectors=None,
                               data_keys=data_keys,
                               xrd_dets=xrd_dets,
                               returns=['data_keys',
                                        'xrd_dets'],
                               repair_method=repair_method)

        # Extract main image_data
        xrd_data = [data_dict[f'{xrd_det}_image']
                    for xrd_det in xrd_dets]

        # Get map shape information. Must follow flying directions
        row_length = scan_md['scan_input'][2]
        col_length = scan_md['scan_input'][5]

        # Parse data_dict
        pos_dict, sclr_dict = {}, {}
        for key, value in data_dict.items():
            if key in sclr_keys and value.ndim == 2:
                sclr_dict[key] = value
            elif key in pos_keys:
                if value.ndim == 2:
                    pos_dict[key] = value
                else:
                    # Check which axis. TEST ME!
                    if key == scan_md['motors'][0]:
                        pos_dict = np.asarray([value], * col_length)
                    elif key == scan_md['motors'][1]:
                        pos_dict[key] = np.asarray([value,] * row_length).T

        # Parse rocking information and add to pos_dict
        # Should there be a unit check?
        if 'energy_energy' in data_dict:
            energy = data_dict['energy_energy']
            wavelength = energy_2_wavelength(energy)
            pos_dict['energy'] = np.asarray([energy,] * row_length).T
            pos_dict['wavelength'] = np.asarray([wavelength,] * row_length).T
        else:
            energy = scan_md.pop('energy')
        if 'nano_stage_th' in data_dict:
            theta = data_dict['nano_stage_th']
            pos_dict['theta'] = np.asarray([theta,] * row_length).T
        else:
            theta = scan_md.pop('theta')

        # Determine null_map
        null_maps = []
        for det in xrd_dets:
            null_key = f'{det}_null_map'
            if null_key in data_dict:
                null_maps.append(np.asarray(data_dict[null_key]))
            else:
                null_maps.append(None)
        
        # Determine filename
        if filename is None:
            basenames = [f"scan{scan_md['scan_id']}",] * len(xrd_dets)
        else:
            if isinstance(filename, list):
                if len(filename) != len(xrd_data):
                    warn_str = ('WARNING: length of specified '
                                + 'filenames does not match the '
                                + 'number of detectors. Naming may '
                                + 'be unexpected.')
                    print(warn_str)
                    basenames = [filename[0],] * len(xrd_dets)
                else:
                    basenames = filename
            else:
                basenames = [filename,] * len(xrd_dets)
        
        # Append discriminators
        filenames = []
        for name, det in zip(basenames, xrd_dets):
            if len(xrd_dets) > 1:
                name += f'_{det}'
            if filename is None:
                name += f'_{cls._hdf_type}'
            if name.split('.')[-1] != 'h5':
                name += '.h5'
            filenames.append(name)

        # if len(xrd_data) > 1:
        #     filenames = [f'scan{scan_md["scan_id"]}_{det}_{self._hdf_type}.h5'
        #                  for det in xrd_dets]
        # else:
        #     filenames = [filename]

        extra_md = {}
        for key in scan_md.keys():
            if key not in ['scan_id',
                           'beamline_id',
                           'energy',
                           'dwell',
                           'theta',
                           'start_time',
                           'scan_input']:
                extra_md[key] = scan_md[key]
        
        xrdslices = []
        for i, (image_data, det) in enumerate(zip(xrd_data, xrd_dets)):
            xds = cls(scan_id=scan_md['scan_id'],
                      wd=wd,
                      filename=filenames[i],
                      save_hdf=save_hdf,
                      image_data=image_data,
                      null_map=null_maps[i],
                      energy=energy,
                      dwell=scan_md['dwell'],
                      theta=theta,
                      sclr_dict=sclr_dict,
                      pos_dict=pos_dict,
                      beamline=scan_md['beamline_id'],
                      facility='NSLS-II',
                      detector=det,
                      scan_input=scan_md['scan_input'],
                      time_stamp=scan_md['time_str'],
                      extra_metadata=extra_md,
                      **kwargs
                      )
            
            # Check for dark-field. Save but do not apply correction.
            if f'{det}_dark' in data_dict:
                note_str = 'Automatic dark-field found'
                if save_hdf:
                    note_str += ' and written to disk'
                note_str += f' for {det}.'
                xds.dark_field = data_dict[f'{det}_dark'] # Could be passed as extra_attrs
                xds.save_images(images='dark_field', units='counts')
                print(note_str)
            
            xrdslices.append(xds)

        print(f'{cls.__name__} loaded!')
        if len(xrdslices) > 1:
            return tuple(xrdslices)
        else:
            # Don't bother returning a tuple or list of xrdslices
            return xrdslices[0]
    

    #################################
    ### Modified Parent Functions ###
    #################################
    
    @copy_docstring(XRDBaseScan.save_current_hdf)
    def save_current_hdf(self, verbose=False):

        super().save_current_hdf(verbose=verbose)

        # Save positions
        if (hasattr(self, 'pos_dict')
            and self.pos_dict is not None):
            # Write to hdf file
            self.save_sclr_pos('positions',
                                self.pos_dict,
                                self.position_units)
        
        # Save spots
        if (hasattr(self, 'spots')
            and self.spots is not None):
            self.save_spots()
        
        # Save vector_map
        if ((hasattr(self, 'vector_map')
             and self.vectors is not None)
            and (hasattr(self, 'edges')
             and self.edges is not None)):
            self.save_vector_map(rewrite_data=True)


    #########################
    ### Utility Functions ###
    #########################

    def _parse_iterable_value(self, value, name):
        """
        Internal function to check values for setattr calls. This is 
        used for the list values (e.g., energy, wavelength, theta).

        Parameters
        ----------
        value : iterable
            Iterable of values to be parsed.
        name : str
            Name of attribute.

        Raises
        ------
        ValueError if length of values does not match the number of
            images.
        TypeError if the value given is not a constant to be iterated
            or an iterable.
        """

        # Fill placeholders if value is nothing
        if (np.any(np.array(value) is None)
            or np.any(np.isnan(value))):
            setattr(self, f'_{name}', np.array([np.nan,]
                                               * self.map_shape[0]))
        # Proceed if object is iterable and not str or dict
        elif (hasattr(value, '__len__')
              and hasattr(value, '__getitem__')
              and not isinstance(value, (str, dict))):
            if len(value) == self.map_shape[0]:
                setattr(self, f'_{name}', np.asarray(value))
            # Assume single value is constant
            elif len(value) == 1:
                setattr(self, f'_{name}', np.array([value[0],]
                                                   * self.map_shape[0]))
            else:
                err_str = (f'{name} must have length '
                           + 'equal to number of rows.')
                raise ValueError(err_str)
        # Assume single value is constant
        elif isinstance(value, (int, float)):
            setattr(self, f'_{name}', np.array([value,]
                                               * self.map_shape[0]))
        else:
            err_str = (f'Unable to handle {name} input. '
                       + 'Provide iterable or value.')
            raise TypeError(err_str)

    
    @copy_docstring(XRDRockingCurve._set_rocking_axis)
    def _set_rocking_axis(self, *args, **kwargs):
        return XRDRockingCurve._set_rocking_axis(self, *args, **kwargs)

    
    ##################
    ### Properties ###
    ##################
    
    @XRDBaseScan.energy.setter
    @copy_docstring(XRDRockingCurve.energy.fset)
    def energy(self, energy):
        return XRDRockingCurve.energy.fset(self, energy)


    @XRDBaseScan.wavelength.setter
    @copy_docstring(XRDRockingCurve.wavelength.fset)
    def wavelength(self, wavelength):
        return XRDRockingCurve.wavelength.fset(self, wavelength)


    @XRDBaseScan.theta.setter
    @copy_docstring(XRDRockingCurve.theta.fset)
    def theta(self, theta):
        return XRDRockingCurve.theta.fset(self, theta)


    @XRDBaseScan.q_arr.getter  
    @copy_docstring(XRDRockingCurve.q_arr.fget)
    def q_arr(self):
        return XRDRockingCurve.q_arr.fget(self)   

    ##############################
    ### Calibrating Map Images ###
    ##############################

    @copy_docstring(XRDMap.integrate1D_map)
    def integrate1D_map(self, *args, **kwargs):
        return XRDMap.integrate1D_map(self, *args, **kwargs)

    
    @copy_docstring(XRDMap.integrate2D_map)
    def integrate2D_map(self, *args, **kwargs):
        return XRDMap.integrate2D_map(self, *args, **kwargs)

    # Backwards compatibility
    integrate1d_map = integrate1D_map
    integrate2d_map = integrate2D_map
        
    
    #######################
    ### Position Arrays ###
    #######################

    def set_positions(self,
                      pos_dict,
                      position_units=None,
                      check_init_sets=False):
        """
        Set the position dictionary attribute.

        Set the internal position dictionary attribute along with the
        position units. If an HDF file is specified, these values will
        be written to the file.

        Parameters
        ----------
        pos_dict : dict of Numpy arrays
            Dictionary of Numpy arrays matching the shape of the first
            two dimensions of images (map shape). Each array is
            associated with the position values along the map by the
            keys.
        position_units : str, optional
            Units of the position measurements. 'μm' by default.
        check_init_sets : bool, optional
            Conditional flag to disable rewriting the position
            information to the HDF file when loading from the HDF file.
            False by default and typical usage.
        """

        # Re-work dictionary keys into stable format
        temp_dict = {}
        for key in list(pos_dict.keys()):
            # Reserved keys...
            if key in ['interp_x', 'interp_y']:
                temp_dict[key] = pos_dict[key]
            # More versatile to include data from multiple sources...
            elif key in ['enc1', '1', 'x', 'X', 'map_x', 'map_X']:
                temp_dict['map_x'] = pos_dict[key]
            elif key in ['enc2', '2' 'y', 'Y', 'map_y', 'map_Y']:
                temp_dict['map_y'] = pos_dict[key]
            # Rocking keys
            elif key in ['energy', 'theta']:
                temp_dict[key] = pos_dict[key]
            elif key in ['angle']:
                temp_dict['theta'] = pos_dict[key]
            elif key in ['wavlength']:
                temp_dict['energy'] = wavelength_2_energy(pos_dict[key])
            else:
                temp_dict[key] = pos_dict[key]
        del pos_dict
        pos_dict = temp_dict

        # Store pos_dict as attribute
        self.pos_dict = pos_dict

        # Set position units
        if position_units is None:
            # default to microns, not that reliable...
            position_units = 'μm' 
        self.position_units = position_units

        # Write to hdf file
        self.save_sclr_pos('positions',
                            self.pos_dict,
                            self.position_units,
                            check_init_sets=check_init_sets)

    
    def map_extent(self,
                   map_x=None,
                   map_y=None,
                   with_step=False):
        """
        Get the map extent for plotting.

        This function will determine the map extent passed to the
        extent keyward argument for plotting images with Matplotlib.
        These values are derived from the internal position dictionary
        if not provided explicitly and are the extents of the positions
        with a half step size appended to each end if requested.

        Parameters
        ----------
        map_x : Numpy.ndarray matching the map shape, optional
            2D array of x-positions with the units of the internally
            stored position units. By default this value will be
            derived from the internal position dictionary.
        map_y : Numpy.ndarray matching the map shape, optional
            2D array of y-positions with the units of the internally
            stored position units. By default this value will be
            derived from the internal position dictionary.
        with_step : bool, optional
            Flag to append the map extends by a half step in both
            directions. This allows the return to be passed directly
            to Matplotlib extent keyward argument for plotting images.

        Returns
        -------
        map_extent : tuple
            Tuple of (min_x, max_x, max_y, min_y) for maps without
            swapped axes, or (min_y, max_y, max_x, min_x) with swapped
            axes. If with_step is True, and half step will be appended
            in both directions for both axes.

        Raises
        ------
        AttributeError if XRDMap is missing the position dictionary or
            'map_x' or 'map_y' is not provided.
        ValueError if map_x or map_y is not provided and cannot be
            interpretted from the position dictionary.
        """

        if ((map_x is None or map_y is None)
             and not hasattr(self, 'pos_dict')):
            err_str = ('XRDMap has no loaded pos_dict.'
                       + '\nPlease load positions or specify '
                       + 'map_x and map_y.')
            raise AttributeError(err_str)

        if map_x is None and hasattr(self, 'pos_dict'):
            # Biased towards interpolated values. More regular
            if _check_dict_key(self.pos_dict, 'interp_x'): 
                map_x = self.pos_dict['interp_x']
            elif _check_dict_key(self.pos_dict, 'map_x'):
                map_x = self.pos_dict['map_x']
            else:
                err_str = ('Cannot find known key '
                           + 'for map_x coordinates.')
                raise ValueError(err_str)
        
        if map_y is None and hasattr(self, 'pos_dict'):
            # Biased towards interpolated values. More regular
            if _check_dict_key(self.pos_dict, 'interp_y'): 
                map_y = self.pos_dict['interp_y']
            elif _check_dict_key(self.pos_dict, 'map_y'):
                map_y = self.pos_dict['map_y']
            else:
                err_str = ('Cannot find known key '
                           + 'for map_y coordinates.')
                raise ValueError(err_str)
        
        x_step, y_step = 0, 0
        # Determine fast scanning direction for map extent
        if (np.mean(np.diff(map_x, axis=1))
            > np.mean(np.diff(map_x, axis=0))):
            # Fast x-axis. Standard orientation.
            #print('Fast x-axis!')
            if with_step:
                x_step = np.mean(np.diff(map_x, axis=1))
                y_step = np.mean(np.diff(map_y, axis=0))
            map_extent = [
                np.mean(map_x[:, 0]) - (x_step / 2),
                np.mean(map_x[:, -1]) + (x_step / 2),
                np.mean(map_y[-1]) + (y_step / 2), # reversed
                np.mean(map_y[0]) - (y_step / 2)# reversed
            ] # [min_x, max_x, max_y, min_y] reversed y for matplotlib
        else: # Fast y-axis. Consider swapping axes???
            if with_step:
                x_step = np.mean(np.diff(map_x, axis=0))
                y_step = np.mean(np.diff(map_y, axis=1))
            map_extent = [
                np.mean(map_y[:, 0]) - (y_step / 2), # reversed
                np.mean(map_y[:, -1]) + (y_step / 2), # reversed
                np.mean(map_x[-1]) + (x_step / 2),
                np.mean(map_x[0]) - (x_step / 2)
            ] # [min_y, max_y, max_x, min_x] reversed x for matplotlib
        
        return tuple(map_extent)



    # TODO: Add interpolation from position dictionary without scan
    # input
    def interpolate_positions(self,
                              scan_input=None,
                              check_init_sets=False):
        """
        Interpolate mapped positions onto a regular grid.

        This method generates new entries in the position dicitionary
        (as "interp_x" and "interp_y") for the nominal mapped positions
        on a regular grid. This method uses the "scan_input" attribute
        to interpolate the nominal mapped positions assuming a
        [xstart, xend, xnum, ystart, yend, ynum] format. New position
        dictionary values are stored internall and written to the HDF
        file if available. 

        Parameters
        ----------
        scan_input : iterable, optional
            Iterable of form [xstart, xend, xnum, ystart, yend, ynum].
            If this parameter is not provided, the internal
            "scan_input" attribute will ber used instead, if available.
        check_init_sets : bool, optional
            Flag whether to overwrite data in the HDF file if available.
            By default this is set to False and should only be True
            when instantiating from the HDF file.

        Raises
        ------
        ValueError if scan_input is not provided and internal attribute
            is not available.
        """

        if scan_input is None:
            if (hasattr(self, 'scan_input')
                and self.scan_input is not None):
                scan_input = self.scan_input
            else:
                err_str = ('Cannot interpolate positiions '
                           + 'without scan input.')
                raise ValueError(err_str)
        
        xstart, xend, xnum = scan_input[0:3]
        interp_x_val = np.linspace(xstart, xend, int(xnum))
        
        ystart, yend, ynum = scan_input[3:6]
        interp_y_val = np.linspace(ystart, yend, int(ynum))

        interp_y, interp_x = np.meshgrid(interp_y_val,
                                         interp_x_val,
                                         indexing='ij')

        if hasattr(self, 'pos_dict') and self.pos_dict is not None:
            self.pos_dict['interp_x'] = interp_x
            self.pos_dict['interp_y'] = interp_y
        
            # Write to hdf file
            self.save_sclr_pos('positions',
                               self.pos_dict,
                               self.position_units,
                               check_init_sets=check_init_sets)
        else:
            warn_str = ('WARNING: No pos_dict found. '
                        + 'Generating from interpolated positions.')
            print(warn_str)
            pos_dict = {
                'interp_x' : interp_x,
                'interp_y' : interp_y
            }
            # Store internally and save    
            self.set_positions(pos_dict,
                               check_init_sets=check_init_sets)
    
    
    #######################
    ### Blobs and Spots ###
    #######################

    @copy_docstring(XRDBaseScan._find_blobs)
    def find_2D_blobs(self, *args, **kwargs):
        super()._find_blobs(*args, **kwargs)

    
    def find_3D_blobs(self, *args, **kwargs):
        raise NotImplementedError
    

    def find_3D_spots(self, *args, **kwargs):
        raise NotImplementedError

    @copy_docstring(XRDRockingCurve.trim_3D_spots)
    def trim_3D_spots(self, *args, **kwargs):
        return XRDRockingCurve.trim_3D_spots(self, *args, **kwargs)

    
    @copy_docstring(XRDMap.pixel_spots)
    def pixel_3D_spots(self, *args, **kwargs):
        return XRDMap.pixel_spots(self, *args, **kwargs)


    @copy_docstring(XRDRockingCurve.save_3D_spots)
    def save_3D_spots(self, *args, **kwargs):
        return XRDRockingCurve.save_3D_spots(self, *args, **kwargs)


    ############################
    ### Vectorizing Map Data ###
    ############################

    @copy_docstring(XRDRockingCurve.get_sampled_edges)
    def get_sampled_edges(self, *args, **kwargs):
        return XRDRockingCurve.get_sampled_edges(self, *args, **kwargs)


    @XRDBaseScan._protect_hdf()
    def vectorize_map_data(self, *args, **kwargs):
        raise NotImplementedError
    

    @XRDBaseScan._protect_hdf()
    def save_vector_map(self, *args, **kwargs):
        raise NotImplementedError

    
    @XRDBaseScan._protect_hdf()
    def load_vector_map(self):
        """
        Load the vector map from the HDF file.
        """

        self._load_vectors(self.hdf)

    #################################
    ### Analysis of Selected Data ###
    #################################
        

    ##########################
    ### Plotting Functions ###
    ##########################
    
    @return_plot_wrapper
    def plot_map(self,
                 map_values,
                 map_extent=None,
                 position_units=None,
                 title=None,
                 fig=None,
                 ax=None,
                 title_scan_id=True,
                 **kwargs):
        """
        Plot a map.

        Map plotting function for any map matching the XRDMap map
        shape.

        Parameters
        ----------
        map_values : 2D Numpy.ndarray matching the map shape
            Values to be plotted as a map.
        map_extent : iterable, optional
            Mapped extent as an iterable with (min_x, max_x, max_y,
            min_y) values. By default the result of the internal
            "map_extent" method.
        position_units : str, optional
            Position units of the map_exent. Internal position units
            are used by default.
        title : str, optional
            Title of the map. By default "Custom Map" will be used.
        fig : Matplotlib Figure instance, optional
            Figure to be used for plotting. Must be given ax. None by
            default, generating an new plot.
        ax : Matplotlib Axes instance, optional
            Axes to be used for plotting. Must be given with fig. None
            by default, generating a new plot.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be prepended with the
            data scan ID. True by default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword agruments, optional
            Other keyword arguments passed to the internal
            Matplotlib.pyplot.imshow function.

        Returns
        -------
        No returns by default. Only returned if return_plot
        keyword argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.

        Raises
        ------
        ValueError if the map_values array does not match the XRDMap
            map shape.
        """
        
        map_values = np.asarray(map_values)
        
        if (hasattr(self, 'pos_dict')
            and (map_values.shape
                   != list(self.pos_dict.values())[0].shape)):
            err_str = (f'Map input shape {map_values.shape} does '
                       + f'not match instance shape '
                       + f'of {list(self.pos_dict.values())[0].shape}')
            raise ValueError(err_str)
        
        if map_extent is None:
            map_extent = self.map_extent(with_step=True)
        if position_units is None:
            position_units = self.position_units

        title = self._title_with_scan_id(
                        title,
                        default_title='Custom Map',
                        title_scan_id=title_scan_id)
        
        return plot_map(map_values,
                        map_extent=map_extent,
                        position_units=position_units,
                        title=title,
                        fig=fig,
                        ax=ax,
                        **kwargs)


    # Interactive plots do not currently accept fig, ax inputs
    @return_plot_wrapper
    def plot_interactive_map(self,
                             data_kw=None,
                             map_kw=None,
                             title_scan_id=True,
                             **kwargs):
        """
        Plot an interactive map of images.

        Plot an interactive map, where each mapped pixel shows its
        associated 2D XRD pattern.

        Parameters
        ----------
        data_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                vmin : float, optional
                    Starting maximum value for dynamic colormap.            
                vmax : float, optional
                    Starting minimum value for dynamic colormap.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                map : 2D Numpy.ndarray matching the map shape, optional
                    Values to be plotted as a map.
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.
        
        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword arguments, optional
            Other keyword arguments passed to interactive plotting
            function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot keyword
        argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if images for plotting dynamic XRD patterns cannot
            be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if data_kw is None:
            data_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(data_kw, 'data'):
            pass
        elif not hasattr(self, 'images'):
            raise ValueError('Could not find images to plot data!')
        elif self.images.ndim != 4:
            err_str = ('XRDData data shape is not 4D, '
                       + f'but {self.images.ndim}.')
            raise ValueError(err_str)
        else:
            data_kw['data'] = self.images

        if (self.corrections['polar_calibration']
            and not _check_dict_key(data_kw, 'x_ticks')):
            if hasattr(self, 'tth') and self.tth is not None:
                data_kw['x_ticks'] = self.tth
                data_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
            if hasattr(self, 'chi') and self.chi is not None:
                data_kw['y_ticks'] = self.chi
                data_kw['x_label'] = ('Azimuthal Angle, χ '
                                     + f'[{self.azimuthal_units}]')

        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_map
            map_kw['title'] = 'Max Detector Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)
        
        return Interactive2DPlot(map_kw,
                                 data_kw,
                                 **kwargs).return_plot()
                    


    @return_plot_wrapper
    def plot_interactive_integration_map(self,
                                         data_kw=None,
                                         map_kw=None,
                                         title_scan_id=True,
                                         **kwargs):
        """
        Plot an interactive map of integrations.

        Plot an interactive map, where each mapped pixel shows its
        associated 1D XRD pattern integration.

        Parameters
        ----------
        data_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                map : 2D Numpy.ndarray matching the map shape, optional
                    Values to be plotted as a map.
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be 
            prepended with the data scan ID. True by 
            default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword arguments, optional
            Other keyword arguments passed to the interactive plotting
            function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot keyword
        argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
            cannot be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if data_kw is None:
            data_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(data_kw, 'data'):
            data_kw['data'] = np.asarray(data_kw['data'])
        elif not hasattr(self, 'integrations'):
            err_str = 'Could not find integrations to plot data!'
            raise ValueError(err_str)
        elif self.integrations.ndim != 3:
            err_str = ('Integration data shape is not 3D, '
                       + f'but {self.integrations.ndim}.')
            raise ValueError(err_str)
        else:
            data_kw['data'] = self.integrations

        if not _check_dict_key(data_kw, 'x_ticks'):
            if hasattr(self, 'tth') and self.tth is not None:
                data_kw['x_ticks'] = self.tth
                data_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
    
        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_integration_map
            map_kw['title'] = 'Max Detector Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)

        return Interactive1DPlot(map_kw,
                                 data_kw,
                                 **kwargs).return_plot()


    @return_plot_wrapper
    def plot_1D_window_sum_map(self,
                               data_kw=None,
                               map_kw=None,
                               title_scan_id=True,
                               **kwargs):
        """
        Plot an interactive map of a 1D window sum.

        Draw a window across the maximum 1D integration and plot a map
        of the sum within this window.

        Parameters
        ----------
        data_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be prepended with the
            data scan ID. True by default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword arguments, optional
            Other keyword arguments passed to the interactive plotting
            function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot keyword
        argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
            cannot be found or are of the wrong shape.
        """
        
        data_kw, map_kw = self._prepare_1D_window_plot(
                                        data_kw=data_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )

        return Interactive1DWindowSumPlot(map_kw,
                                          data_kw,
                                          **kwargs).return_plot()
        

    @return_plot_wrapper
    def plot_1D_window_com_map(self,
                               data_kw=None,
                               map_kw=None,
                               title_scan_id=True,
                               **kwargs):
        """
        Plot an interactive map of a 1D window center of mass.

        Draw a window across the maximum 1D integration and plot a map
        of the center of mass within this window.

        Parameters
        ----------
        data_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                y_min : float, optional
                    Starting maximum y-value (intensity) for dynamic
                    plot.            
                y_max : float, optional
                    Starting minimum y-value (intensity) for dynamic
                    plot.
                x_min : float, optional
                    Starting maximum x-value (scattering angle) for
                    dynamic plot.            
                x_max : float, optional
                    Starting minimum x-value (scattering angle) for
                    dynamic plot.

        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.

            Useful Keywords:
                title : str, optional
                    Title of map plot.
                vmin : float, optional
                    Starting minimum value for map colormap.
                vmax : float, optional
                    Starting maximum value for map colormap.

        title_scan_id : bool, optional
            Flag dictating if the plot title will be prepended with the
            data scan ID. True by default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword arguments, optional
            Other keyword arguments passed to the interactive plotting
            function.
        
        Returns
        -------
        No returns by default. Only returned if return_plot keyword
        argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.
        
        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
            cannot be found or are of the wrong shape.
        """

        data_kw, map_kw = self._prepare_1D_window_plot(
                                        data_kw=data_kw,
                                        map_kw=map_kw,
                                        title_scan_id=title_scan_id
                                        )

        return Interactive1DWindowCOMPlot(map_kw,
                                          data_kw,
                                          **kwargs).return_plot()


    def _prepare_1D_window_plot(self,
                                data_kw=None,
                                map_kw=None,
                                title_scan_id=True):
        """
        Internal function for preparing 1D windowed plots.

        Parameters
        ----------
        data_kw : dict, optional
            Dictionary of keyword arguments passed for dynamic
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        map_kw : dict, optional
            Dictionary of keyword arguments passed for the map
            plotting. Empty by default, and the values will be filled
            with the relevant instance parameters.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be prepended with the
            data scan ID. True by default.

        Raises
        ------
        ValueError if integrations for plotting dynamic XRD patterns
            cannot be found or are of the wrong shape.
        """
        
        # Python doesn't play well with mutable default kwargs
        if data_kw is None:
            data_kw = {}
        if map_kw is None:
            map_kw = {}

        if _check_dict_key(data_kw, 'data'):
            data_kw['data'] = np.asarray(data_kw['data'])
        elif not hasattr(self, 'integrations'):
            err_str = 'Could not find integrations to plot data!'
            raise ValueError(err_str)
        elif self.integrations.ndim != 3:
            err_str = ('Integration data shape is not 3D, '
                       + f'but {self.integrations.ndim}.')
            raise ValueError(err_str)
        else:
            data_kw['data'] = self.integrations

        if not _check_dict_key(data_kw, 'x_ticks'):
            if hasattr(self, 'tth') and self.tth is not None:
                data_kw['x_ticks'] = self.tth
                data_kw['x_label'] = ('Scattering Angle, 2θ '
                                     + f'[{self.scattering_units}]')
    
        # Add default map_kw information if not already included
        if not _check_dict_key(map_kw, 'map'):
            map_kw['map'] = self.max_integration_map
            map_kw['title'] = 'Max Integration Intensity'
        if not _check_dict_key(map_kw, 'x_ticks'):
            map_kw['x_ticks'] = np.round(np.linspace(
                *self.map_extent()[:2], # without step
                self.map_shape[1]), 2)
        if not _check_dict_key(map_kw, 'y_ticks'):
            map_kw['y_ticks'] = np.round(np.linspace(
                *self.map_extent()[2:], # without step
                self.map_shape[0]), 2)
        if hasattr(self, 'position_units'):
            if not _check_dict_key(map_kw, 'x_label'):
                map_kw['x_label'] = ('x position '
                                     + f'[{self.position_units}]')
            if not _check_dict_key(map_kw, 'y_label'):
                map_kw['y_label'] = ('y position '
                                     + f'[{self.position_units}]')
        
        if 'title' not in map_kw:
            map_kw['title'] = None
        map_kw['title'] = self._title_with_scan_id(
                            map_kw['title'],
                            default_title='Custom Map',
                            title_scan_id=title_scan_id)
        if 'title' not in data_kw:
            data_kw['title'] = None
        data_kw['title'] = self._title_with_scan_id(
                            data_kw['title'],
                            default_title='',
                            title_scan_id=title_scan_id)    
        
        return data_kw, map_kw
    

    @return_plot_wrapper
    def plot_waterfall(self, **kwargs):
        
        """
        Generate waterfall plot integrated along a map axis.

        Sum all integrations along a mapped axis and plot the results
        as a waterfall plot.

        Parameters
        ----------
        axis : int, optional
            Axis for integrating the data. Only 0 and 1 are allowed.
            0 by default.
        integration_method : str, optional
            Method for integrating data. Accepts 'sum' or 'max' along
            the axis of integration. 'max' by default.
        v_offset : float, optional
            Relative vertical offset value of each integration. 0 will
            have no offset, 1 will offset each integration by the
            maximum integration range preventing any overlap. 0.25 by
            default.
        tth : iterable, optional
            Two theta scattering angle of the integration data. Must
            have the same length as the integration data. None by
            default and will look for an internal tth attribute, or use
            a simple range if unavaible.
        units : str, optional
            Units of two theta scattering angle. None by default and
            will use an internal scattering_angle attribute if
            available.
        cmap : str, optional
            Colormap for distinguishing integrations. None by default
            and all integrations will be black.
        fig : Matplotlib Figure instance, optional
            Figure to be used for plotting. Must be given ax. None by
            default, generating an new plot.
        ax : Matplotlib Axes instance, optional
            Axes to be used for plotting. Must be given with fig. None
            by default, generating a new plot.
        title : str, optional
            Title of plot. By default, a title will be generated from
            the waterfall plot.
        title_scan_id : bool, optional
            Flag dictating if the plot title will be prepended with the
            data scan ID. True by default.
        return_plot : bool, optional
            Flag to determine if figure and axes option are returned.
            False be default with no returns.
        kwargs : keyword arguments, optional
            Other keyword arguments passed to the internal
            Matplotlib.pyplot.plot function.

        Returns
        -------
        No returns by default. Only returned if return_plot keyword
        argument is True.

        fig : Matplotlib Figure instance
            Figure of plot.
        ax : Matplotlib Axes instance, optional
            Individual axes of plot.

        Raises
        ------
        AttributeError for data without integrations.
        ValueError if the integration method is not
            acceptable.
        """

        if 'axis' in kwargs:
            axis = kwargs.pop('axis')
        else:
            axis = 0 # default
        
        if axis == 0:
            axis_text = 'Horizontal'
        elif axis == 1:
            axis_text = 'Vertical'

        return self._plot_waterfall(axis=axis,
                                    axis_text=axis_text,
                                    **kwargs)