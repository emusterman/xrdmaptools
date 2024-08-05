# This entire module is to separate the make_xrdmap_hdf to avoid circular imports
# I could not figure another way around this...


# Local imports
from ..XRDMap import XRDMap

# A very convenient wrapper without returning the class
def make_xrdmap_hdf(scanid=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None,
                    repair_method='fill',
                    return_xrdmap=False):
    
    print('*' * 72)
    xrdmap = XRDMap.from_db(
                scanid=scanid,
                broker=broker,
                filedir=filedir,
                filename=filename,
                poni_file=poni_file,
                save_hdf=True,
                repair_method=repair_method
                   )
    print('*' * 72)

    if return_xrdmap: 
        # Just XRDMap.from_db at this point
        return xrdmap
    

