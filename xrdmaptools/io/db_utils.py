# This entire module is to separate the make_xrdmap_hdf to avoid circular imports
# I could not figure another way around this...


# Local imports
from ..XRDMap import XRDMap

# A very convenient wrapper without returning the class
def make_xrdmap_hdf(scan_id=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None,
                    repair_method='fill'):
    
    print('*' * 72)
    xrdmap = XRDMap.from_db(
                scan_id=scan_id,
                broker=broker,
                filedir=filedir,
                filename=filename,
                poni_file=poni_file,
                save_hdf=True,
                repair_method=repair_method,
                dask_enabled=True # Hard-coded to allow lazy loading
                   )
    print('*' * 72)
    

