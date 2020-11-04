import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import date
import utils.par_checker as pc

# Read tim/par files
import pint.toa as toa
import pint.models as models
import pint.residuals
from pint.modelutils import model_equatorial_to_ecliptic

def write_par(fitter,addext=''):
    """Writes a timing model object to a par file in the working directory.

    Parameters
    ==========
    fitter: `pint.fitter` object
    """
    source = fitter.get_allparams()['PSR']
    date_str = date.today().strftime('%Y%m%d')
    outfile = '%s_PINT_%s%s.par' % (source,date_str,addext)

    fout=open(outfile,'w')
    fout.write(fitter.model.as_parfile())
    fout.close()


def write_include_tim(source,tim_file_list):
    """Writes file listing tim files to load as one PINT toa object (using INCLUDE).

    Parameters
    ==========
    source: string
        pulsar name
    tim_file_list: list
        tim files to include

    Returns
    =======
    out_tim: tim filename string
    """
    out_tim = '%s.tim' % (source)
    f = open(out_tim,'w')

    for tf in tim_file_list:
        f.write('INCLUDE %s\n' % (tf))

    f.close()
    return out_tim

def plot_res(fitter,restype='prefit'):
    """Simple plotter for prefit/postfit residuals.

    Parameters
    ==========
    fitter: `pint.fitter` object
    restype: string, optional
        controls type of residuals plotted, prefit/postfit

    Raises
    ======
    ValueError
        If restype is not recognized.
    """
    fo = fitter
    obslist = list(fitter.toas.observatories)

    # Select toas from each observatory.
    for obso in obslist:

        select_array = (fitter.toas.get_obss()==obso)
        fitter.toas.select(select_array)

        if 'pre' in restype:
            plt.errorbar(
            fitter.toas.get_mjds(), fitter.resids_init.time_resids.to(u.us)[select_array], fitter.toas.get_errors().to(u.us), fmt="x",label=obso
            )
            restype_str = 'Pre'
        elif 'post' in restype:
            plt.errorbar(
            fitter.toas.get_mjds(), fitter.resids.time_resids.to(u.us)[select_array], fitter.toas.get_errors().to(u.us), fmt="x", label=obso
            )
            restype_str = 'Post'
        else:
            raise ValueError("Residual type (%s) not recognized. Try prefit/postfit." % (restype)) 

        plt.title("%s %s-Fit Timing Residuals" % (fitter.model.PSR.value,restype_str))
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.grid()
        plt.legend()

        fitter.toas.unselect()

def center_epochs(model,toas):
    """Center PEPOCH (POSEPOCH, DMEPOCH) using min/max TOA values.

    Parameters
    ==========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object

    Returns
    =======
    model: `pint.model.TimingModel` object
        with centered epoch(s)
    """
    midmjd=(toas.get_mjds().value.max()+toas.get_mjds().value.min())/2.
    model.change_pepoch(midmjd)

    try:
        model.change_posepoch(midmjd)
    except:
        pass

    try:
        model.change_dmepoch(midmjd)
    except:
        pass

    return model

def apply_snr_cut(toas,snr_cut,summary=False):
    """Imposes desired signal-to-noise ratio cut

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    snr_cut: float
        selects TOAs with snr > snr_cut
    summary: boolean, optional
        print toa summary
    """
    # Might want a warning here. SNR cut should happen before others for intended effect. 
    # toas.unselect()
    toas.select((np.array(toas.get_flag_value('snr')) > snr_cut)[0])
    
    if summary:    
        toas.print_summary()

def apply_mjd_cut(toas,configDict,summary=False):
    """Imposes cuts based on desired start/stop times (MJD)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    configDict: Dict
        configuration parameters read directly from yaml
    summary: boolean, optional
        print toa summary
    """

    if configDict['ignore']['mjd-start']:
        mjd_min = configDict['ignore']['mjd-start']
        select_min = (toas.get_mjds() > mjd_min*u.d)
    else:
        select_min = np.array([True]*len(toas))


    if configDict['ignore']['mjd-end']:
        mjd_max = configDict['ignore']['mjd-end']
        select_max = (toas.get_mjds() < mjd_max*u.d)
    else:
        select_max = np.array([True]*len(toas))

    toas.select(select_min & select_max)

    if summary:
        toas.print_summary()

def load_and_check(configDict,usepickle=False):
    """Loads toas/model objects using configuration info, runs basic checks.

    Checks that ephem and bipm_version have been set to the latest available versions; checks
    for equatorial astrometric parameters (converts to ecliptic, if necessary); also checks
    source name, and for appropriate number of jumps. Checks are functions from par_checker.py.

    Parameters
    ==========
    configDict: Dict
        configuration parameters read directly from yaml
    usepickle: boolean, optional
        produce TOA pickle object

    Returns
    =======
    to: `pint.toa.TOAs` object
        passes all checks
    mo: `pint.model.TimingModel` object
        passes all checks
    """
    source = configDict['source']
    tim_path = configDict['tim-directory']
    tim_files = [tim_path+tf for tf in configDict['toas']]
    tim_filename = write_include_tim(source,tim_files)
    to = toa.get_TOAs(tim_filename, usepickle=usepickle, bipm_version=configDict['bipm'], ephem=configDict['ephem'])

    # Check ephem/bipm
    pc.check_ephem(to)
    pc.check_bipm(to)

    # Identify receivers present
    receivers = set(to.get_flag_value('fe')[0])

    # Load the timing model
    par_path = configDict['par-directory']
    mo = models.get_model(par_path+configDict['timing-model'])

    # Convert to/add AstrometryEcliptic component model if necessary.
    if 'AstrometryEquatorial' in mo.components:
        model_equatorial_to_ecliptic(mo)

    # Basic checks on timing model
    pc.check_name(mo)
    pc.check_jumps(mo,receivers)

    return to, mo

def check_fit(fitter):
    """Check that pertinent parameters are unfrozen.

    Note: process of doing this robustly for binary models is not yet automated. Checks are
    functions from par_checker.py.

    Parameters
    ==========
    fitter: `pint.fitter` object 
    """
    pc.check_spin(fitter.model)
    pc.check_astrometry(fitter.model)
