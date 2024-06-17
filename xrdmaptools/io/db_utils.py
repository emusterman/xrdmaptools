# This entire module is to separate the make_crdmap_hdf to avoid circular imports
# I could not figure another way around this...


# Local imports
from ..XRDMap import XRDMap

# Just a convenience wrapper wihout returning the class
def make_xrdmap_hdf(scanid=-1,
                    broker='manual',
                    filedir=None,
                    filename=None,
                    poni_file=None,
                    repair_method='replace'):
    
    print('*' * 72)
    XRDMap.from_db(scanid=scanid,
                   broker=broker,
                   filedir=filedir,
                   filename=filename,
                   poni_file=poni_file,
                   save_hdf=True,
                   repair_method=repair_method)
    print('*' * 72)
    



def make_xrdmap_composite():
    raise NotImplementedError()
