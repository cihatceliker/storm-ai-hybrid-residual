# Standard library imports
import time
from datetime import datetime, timedelta
from math import radians

# Data processing imports
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# Orekit imports
import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, PositionAngleType
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates, IERSConventions
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory, CelestialBody
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.models.earth.atmosphere import DTM2000, PythonAtmosphere
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Only keep what's needed for file handling
from orekit import JArray

# MSIS model
from pymsis import msis

# Custom utilities
from util import orbit_mean
from feat_eng import FORECAST_SIZE
from tqdm import tqdm

# Initialize Orekit JVM
vm = orekit.initVM()
setup_orekit_curdir()

# Global constants
MSIS_INPUT_DATA = MarshallSolarActivityFutureEstimation(
    MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
    MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
)
print(MSIS_INPUT_DATA.getMinDate(), MSIS_INPUT_DATA.getMaxDate())

EME2000 = FramesFactory.getEME2000()
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
EARTH = OneAxisEllipsoid(Constants.IERS2010_EARTH_EQUATORIAL_RADIUS,
                         Constants.IERS2010_EARTH_FLATTENING, ITRF)
MU = Constants.IERS2010_EARTH_MU
UTC = TimeScalesFactory.getUTC()
SUN = CelestialBodyFactory.getSun()




def create_physics_features(initial_states, omni_df):
    """Generate physics-based atmospheric density features from multiple atmospheric models.
    
    Computes density estimates using MSIS, DTM2000, and JB2008 models for each state,
    then calculates logarithmic values and derived features.
    
    Args:
        initial_states: DataFrame containing initial orbital elements
        omni_df: DataFrame with space weather parameters
    
    Returns:
        DataFrame with physics-based features for each state
    """
    physics = []
    
    # Process each initial state
    for i, row in tqdm(initial_states.iterrows(), total=initial_states.shape[0]):
        try:
            # Get density estimates from different atmospheric models
            orbit_means = get_physics_estimates(row.to_dict(), 
                                               omni_df[omni_df["file_id"]==i].reset_index(drop=True))
            
            # Create dataframe with timestamps
            ph = pd.DataFrame(orbit_means).assign(file_id=i)
            ph["Timestamp"] = sorted([row["Timestamp"]-timedelta(minutes=10*f) 
                                     for f in range(FORECAST_SIZE)])
            physics.append(ph)
        except Exception as e:
            print(e, i)
    
    # Combine all results
    physics = pd.concat(physics, ignore_index=True)
    
    # Apply log transform to density values
    physics["msis"] = np.log(physics["msis"])
    physics["dtm"] = np.log(physics["dtm"])
    
    # Calculate local solar time dependent features
    calc_lst = lambda x1, x2: ((physics[x1] - physics[x2]) * 
                          np.sin(2 * np.pi * physics["Timestamp"].dt.hour / 24.0))
    physics["dtm_msis_lst"] = calc_lst("dtm", "msis")
    
    # Calculate differences between models
    physics["msis_dtm"] = physics["msis"] - physics["dtm"]
    
    # Calculate ratios between models
    physics["msis_dtm_ratio"] = physics["msis"] / physics["dtm"]
    
    return physics

# Unified custom atmosphere model
class CustomAtmosphere(PythonAtmosphere):
    """Custom atmosphere model for DTM2000."""
    def __init__(self, input_data, sun, earth):
        super().__init__()
        self.atm = DTM2000(input_data, sun, earth)
        self.earth = earth
        
    def getDensity(self, date, position, frame):
        return self.atm.getDensity(date, position, frame)
    
    def getVelocity(self, date, position, frame):
        bodyToFrame = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
        posInBody = bodyToFrame.getStaticInverse().transformPosition(position)
        pv_body = PVCoordinates(posInBody, Vector3D.ZERO)
        pvFrame = bodyToFrame.transformOnlyPV(pv_body)
        return pvFrame.getVelocity()


# Helper functions
def calc_ap(omni_data):
    """Build the 7-element Ap array for NRLMSIS from hourly OMNI2 data."""
    idx = omni_data.index[-1]
    ap_daily = omni_data.loc[idx, 'ap_index_nT']
    
    ap_3hr = [omni_data.iloc[idx - k]['ap_index_nT'] if (idx - k) >= 0 else ap_daily
              for k in (0, 3, 6, 9)]
    
    ap_12_33 = omni_data.iloc[max(idx-33, 0):max(idx-11, 0):3]['ap_index_nT'].mean()
    ap_36_57 = omni_data.iloc[max(idx-57, 0):max(idx-35, 0):3]['ap_index_nT'].mean()
    
    return [ap_daily, *ap_3hr, ap_12_33, ap_36_57]


# Main propagation functions
def propagate_msis(initial_row, omni2, step=600, horizon=3*86400):
    """Propagate orbit using MSIS atmospheric model."""
    ts = initial_row['Timestamp']
    epoch = AbsoluteDate(ts.year, ts.month, ts.day,
                        ts.hour, ts.minute, float(ts.second), UTC)
    orb = KeplerianOrbit(initial_row['Semi-major Axis (km)']*1e3,
                        initial_row['Eccentricity'],
                        radians(initial_row['Inclination (deg)']),
                        radians(initial_row['Argument of Perigee (deg)']),
                        radians(initial_row['RAAN (deg)']),
                        radians(initial_row['True Anomaly (deg)']),
                        PositionAngleType.TRUE, EME2000, epoch, MU)

    n = horizon // step
    adates = [epoch.shiftedBy(float(k*step)) for k in range(n)]
    lla = np.empty((n, 3))
    for k, ad in enumerate(adates):
        geod = EARTH.transform(orb.shiftedBy(float(k*step)).getPVCoordinates().getPosition(),
                              EME2000, ad)
        lla[k] = [np.rad2deg(geod.getLatitude()),
                 np.rad2deg(geod.getLongitude()),
                 geod.getAltitude()/1e3]  # km

    # Use space weather parameters from input data
    sw = omni2.iloc[-1]
    f107 = sw['f10.7_index']
    f107_81day = omni2['f10.7_index'].mean()
    ap7 = calc_ap(omni2.reset_index(drop=True))
    
    #storm = -1 if ap7[1] >= 50 else None
    storm = -1 if sw["geomagnetic_storm"] == 1 else None
    adates = np.array([np.datetime64(str(ad.toString())) for ad in adates])
    rho = msis.run(adates, lla[:,1], lla[:,0], lla[:,2], 
                  f107s=np.full(n, f107), 
                  f107as=np.full(n, f107_81day), 
                  aps=np.tile(ap7, (n,1)), 
                  geomagnetic_activity=storm)
    return rho[:,0]


def prop_orbit(initial_state, CustomAtmosphereClass, duration=3*86400.0, step=600.0):
    """Propagate orbit with specified atmosphere model."""
    ts = initial_state["Timestamp"]
    date = AbsoluteDate(ts.year, ts.month, ts.day, ts.hour, ts.minute, 00.0000, UTC)
    
    # Initialize orbit parameters
    semi_major_axis = initial_state['Semi-major Axis (km)'] * 1e3
    eccentricity = initial_state['Eccentricity']
    inclination = radians(initial_state['Inclination (deg)'])
    raan = radians(initial_state['RAAN (deg)'])
    arg_perigee = radians(initial_state['Argument of Perigee (deg)'])
    true_anomaly = radians(initial_state['True Anomaly (deg)'])
    
    # Create initial orbit and convert to equinoctial
    initial_orbit = KeplerianOrbit(semi_major_axis, eccentricity, inclination,
                                 arg_perigee, raan, true_anomaly,
                                 PositionAngleType.TRUE, EME2000, date, MU)
    initial_orbit = EquinoctialOrbit(initial_orbit)
    
    # Initialize state
    initialState = SpacecraftState(initial_orbit, 260.0)  # 260.0 kg spacecraft mass
    
    # Create propagated states list
    states = [initialState.shiftedBy(float(dt)) for dt in np.arange(0.0, duration+1e-3, step)]
    
    # Initialize atmosphere model using global SUN constant
    atmosphere = CustomAtmosphereClass(MSIS_INPUT_DATA, SUN, EARTH)
    
    # Calculate densities along the trajectory
    densities = []
    for state in states:
        density = atmosphere.getDensity(state.getDate(), state.getPVCoordinates().getPosition(), state.getFrame())
        densities.append(density)
    
    return states, densities


def get_physics_estimates(state, omni):
    """Compare density estimates from different atmospheric models."""
    # Create a copy of the state to avoid modifying the original
    state_copy = state.copy()
    
    # DTM2000 model
    _, densities_dtm = prop_orbit(state_copy, 
                                lambda cswl, sun, earth: CustomAtmosphere(MSIS_INPUT_DATA, sun, earth), 
                                duration=3*86400.0, 
                                step=600.0)
    dtm = orbit_mean(densities_dtm[:FORECAST_SIZE], state_copy["Semi-major Axis (km)"])

    # MSIS model - adjusting timestamp
    state_copy["Timestamp"] -= timedelta(minutes=1)
    msis = propagate_msis(state_copy, omni)

    return {"msis": orbit_mean(msis, state_copy["Semi-major Axis (km)"]), "dtm": dtm}
