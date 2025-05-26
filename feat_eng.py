import pandas as pd
import numpy as np
from datetime import datetime, timedelta


OMNI_SEQUENCE_SIZE = 24 * 60
FORECAST_SIZE = 432

# these features will be transformed accordingly. shift to positive -> log
OMNI_LOGGED = ['BZ_nT_GSE', 'BZ_nT_GSM', 'Dst_index_nT', 'pc_index', 'sigma_theta_V_degrees', 
               'Alfen_mach_number', 'f10.7_change', 'hours_since_storm', 'ap_index_nT', 
               'Alpha_Prot_ratio', 'RMS_field_vector_nT', 'MS_Mach_South', 
               'SW_Proton_Density_N_cm3', 'RMS_BX_GSE_nT', 'sigma_V_km_s', 
               'RMS_ratio_field_to_mag', 'speed_sigma_ratio', 'BZ_southward', 
               'RMS_BY_GSE_nT', 'Dst_change_6h', 'E_electric_field', 'RMS_BZ_GSE_nT', 
               'sigma_phi_V_degrees', 'temp_sigma_ratio', 'Flow_pressure', 'SW_ram_pressure', 
               'rectified_E_field', 'RMS_magnitude_nT', 'SW_Plasma_flow_lat_angle_cos', 
               'sigma_n_N_cm3', 'SW_Plasma_Temperature_K', 'SW_Plasma_flow_long_angle_cos', 
               'Quasy_Invariant', 'sigma_ratio', 'Proton_flux_>10_Mev', 'Proton_flux_>30_Mev', 
               'Proton_flux_>60_Mev', 'sigma_T_K', 'energy_coupling', 
               'Kp_index', 'f10.7_index', 'AE_index_nT', 'AL_index_nT', 'AU_index_nT']


def add_time_features(df):
    """Add time-based features relevant to atmospheric density"""
    
    # Extract hour and calculate diurnal effect
    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Calculate day of year (seasonal effect)
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    return df



def feat_eng_physics(x):
    def feat_eng_physics_single(x, col):
        feats = {
            col+"_mean_log": np.mean(np.exp(x[col])), 
            col+"_mean": np.mean(x[col]), 
            col+"_std": np.std(x[col]), 
            col+"_min": np.min(x[col]), 
            col+"_max": np.max(x[col]), 
            col+"_range": np.max(x[col])-np.min(x[col]), 
            col+"_skew": x[col].skew(), 
            col+"_kurt": x[col].kurtosis(), 
        }
        fft = np.fft.rfft(x[col], n=FORECAST_SIZE)
        amps = np.abs(fft)[1:5]
        phs  = np.angle(fft)[1:5]
        fft_feats = np.concatenate([amps,phs])
        feats.update({f"{col}_fft_{i}": f for i, f in enumerate(fft_feats)})
        feats.update({f"{col}_quantile_{f}": np.quantile(x[col], f) for f in [0.1,0.25,0.5,0.75,0.9]})
        return feats
    
    feats = [x["msis"].corr(x["dtm"])]
    feats = {f"corr_{i}": f for i, f in enumerate(feats)}
    for ph_col in ["msis","dtm"]:
        feats.update(feat_eng_physics_single(x, ph_col))
    return pd.Series(feats)


def statics_from_omni(initial_states, omni_df):
    # from omni to static
    initial_states["omni_total_missing"] = omni_df.groupby("file_id")["total_missing"].first()
    index_cols = ['Kp_index', 'Dst_index_nT', 'ap_index_nT', 
                  'f10.7_index', 'AE_index_nT', 'AL_index_nT', 'AU_index_nT']
    index_cols += [f+"_logged" for f in index_cols if f in OMNI_LOGGED]
    print(index_cols)
    for col in index_cols:
        a = omni_df.groupby("file_id")[col].last()
        initial_states[col] = a
    
    buckets = {
        "Altitude (km)": [291.501,317.017,342.532,368.048,393.564,419.079,444.595,470.11,495.626],
        'Kp_index': [3.0, 7.0, 10.0, 13.0, 20.0, 23.0, 30.0, 37.0],
        'Dst_index_nT': [-31.0, -22.0, -16.0, -12.0, -9.0, -5.0, -3.0, 1.0, 5.0],
        'ap_index_nT': [2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 15.0, 22.0],
        'f10.7_index': [68.3, 69.6, 71.3, 74.9, 80.6, 90.4, 105.2, 123.9, 151.42],
        'AE_index_nT': [25.0, 36.0, 48.0, 66.0, 90.0, 110.0, 168.0, 262.0, 415.0],
        'AL_index_nT': [-278.0,-165.0,-102.0,-58.0,-46.0,-32.0,-22.0,-15.0,-9.0],
        'AU_index_nT': [11.0, 17.0, 23.0, 30.0, 40.0, 47.0, 67.0, 96.0, 144.0]
    }
    for col, bins in buckets.items():
        bins = [-np.inf, *bins, np.inf]
        initial_states = initial_states.copy()
        for i, (v1, v2) in enumerate(zip(bins[:-1], bins[1:])):
            initial_states[f"is_{col}_cat_{i}"] = ((initial_states[col]>=v1) & 
                                                   (initial_states[col]<v2)).astype(int)
    
    initial_states["Flux_FLAG"] = omni_df.groupby("file_id")["Flux_FLAG"].mean()
    
    initial_states["last3_days_ap_mean"] = omni_df.groupby("file_id")["ap_index_nT"].apply(lambda x: x.tail(3*24).mean())
    initial_states["last3_days_ap_max"] = omni_df.groupby("file_id")["ap_index_nT"].apply(lambda x: x.tail(3*24).max())
    initial_states["last3_days_ap_std"] = omni_df.groupby("file_id")["ap_index_nT"].apply(lambda x: x.tail(3*24).std())
    
    a = omni_df.groupby("file_id")["ID_for_IMF_spacecraft"].first()
    initial_states["is_ID_for_IMF_spacecraft_51"] = (51==a).astype(int)
    initial_states["is_ID_for_IMF_spacecraft_71"] = (71==a).astype(int)

    a = omni_df.groupby("file_id")["ID_for_SW_Plasma_spacecraft"].first()
    for val in [52,71,51,99,50,60]:
        initial_states[f"is_ID_for_SW_Plasma_spacecraft_{val}"] = (val==a).astype(int)

    omni_df.drop(columns=["total_missing","ID_for_IMF_spacecraft","ID_for_SW_Plasma_spacecraft"], 
                 inplace=True)
    return initial_states, omni_df


def read_process_omni_file(omni_file_path, file_id):
    omni_nans = {
        'AE_index_nT': 9999.0,
        'AL_index_nT': 99999.0,
        'AU_index_nT': 99999.0,
        'Alfen_mach_number': 999.9,
        'Alpha_Prot_ratio': 9.999,
        'BX_nT_GSE_GSM': 999.9,
        'BY_nT_GSE': 999.9,
        'BY_nT_GSM': 999.9,
        'BZ_nT_GSE': 999.9,
        'BZ_nT_GSM': 999.9,
        'E_electric_field': 999.99,
        'Flow_pressure': 99.99,
        'Lat_Angle_of_B_GSE': 999.9,
        'Long_Angle_of_B_GSE': 999.9,
        'Magnetosonic_Mach_number': 99.9,
        'Plasma_Beta': 999.99,
        'Proton_flux_>10_Mev': 99999.99,
        'Proton_flux_>30_Mev': 99999.99,
        'Proton_flux_>60_Mev': 99999.99,
        'Quasy_Invariant': 9.9999,
        'RMS_BX_GSE_nT': 999.9,
        'RMS_BY_GSE_nT': 999.9,
        'RMS_BZ_GSE_nT': 999.9,
        'RMS_field_vector_nT': 999.9,
        'RMS_magnitude_nT': 999.9,
        'SW_Plasma_Speed_km_s': 9999.0,
        'SW_Plasma_Temperature_K': 9999999.0,
        'SW_Plasma_flow_lat_angle': 999.9,
        'SW_Plasma_flow_long_angle': 999.9,
        'SW_Proton_Density_N_cm3': 999.9,
        'Scalar_B_nT': 999.9,
        'Vector_B_Magnitude_nT': 999.9,
        'f10.7_index': 999.9,
        'num_points_IMF_averages': 999.0,
        'num_points_Plasma_averages': 999.0,
        'pc_index': 999.9,
        'sigma_T_K': 9999999.0,
        'sigma_V_km_s': 9999.0,
        'sigma_n_N_cm3': 999.9,
        'sigma_phi_V_degrees': 999.9,
        'sigma_ratio': 9.999,
        'sigma_theta_V_degrees': 999.9
    }
    
    df = pd.read_csv(omni_file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    for k, v in omni_nans.items():
        df[k] = df[k].replace(v, np.nan)
    
    df.drop(columns=["YEAR","DOY","Hour","Bartels_rotation_number"], inplace=True)
    df.sort_values(["Timestamp"], inplace=True, ignore_index=True)
    
    total_missing = (
        df.isna().sum().sum() + 
        (OMNI_SEQUENCE_SIZE - df.shape[0]) * df.shape[1]
    ) / (df.shape[1] * OMNI_SEQUENCE_SIZE)
    
    df = df.set_index("Timestamp").apply(lambda x: x.interpolate().ffill().bfill())
    df = df.reset_index()
    
    if df.shape[0] == 0:
        print("edge case in omni", file_id)
        # making it one row
        df['Timestamp'] = pd.to_datetime(["2010-12-11 12:00:00"], format='%Y-%m-%d %H:%M:%S')
        total_missing = 1
    
    fill = {
        'AE_index_nT': 90.0,
        'AL_index_nT': -46.0,
        'AU_index_nT': 40.0,
        'Alfen_mach_number': 9.4,
        'Alpha_Prot_ratio': 0.029,
        'BX_nT_GSE_GSM': 0.0,
        'BY_nT_GSE': -0.1,
        'BY_nT_GSM': -0.1,
        'BZ_nT_GSE': 0.0,
        'BZ_nT_GSM': 0.0,
        'Dst_index_nT': -9.0,
        'E_electric_field': -0.0,
        'Flow_pressure': 1.55,
        'Flux_FLAG': -1.0,
        'ID_for_IMF_spacecraft': 51.0,
        'ID_for_SW_Plasma_spacecraft': 52.0,
        'Kp_index': 13.0,
        'Lat_Angle_of_B_GSE': -0.1,
        'Long_Angle_of_B_GSE': 183.2,
        'Lyman_alpha': 0.006503,
        'Magnetosonic_Mach_number': 5.8,
        'Plasma_Beta': 1.7,
        'Proton_flux_>10_Mev': 0.22,
        'Proton_flux_>30_Mev': 0.13,
        'Proton_flux_>60_Mev': 0.09,
        'Quasy_Invariant': 0.0134,
        'RMS_BX_GSE_nT': 0.8,
        'RMS_BY_GSE_nT': 0.9,
        'RMS_BZ_GSE_nT': 1.0,
        'RMS_field_vector_nT': 1.7,
        'RMS_magnitude_nT': 0.2,
        'R_Sunspot_No': 26.0,
        'SW_Plasma_Speed_km_s': 404.0,
        'SW_Plasma_Temperature_K': 64690.0,
        'SW_Plasma_flow_lat_angle': -0.8,
        'SW_Plasma_flow_long_angle': -0.2,
        'SW_Proton_Density_N_cm3': 4.7,
        'Scalar_B_nT': 4.8,
        'Vector_B_Magnitude_nT': 4.2,
        'ap_index_nT': 5.0,
        'f10.7_index': 80.9,
        'num_points_IMF_averages': 59.0,
        'num_points_Plasma_averages': 35.0,
        'pc_index': 0.6,
        'sigma_T_K': 8841.0,
        'sigma_V_km_s': 5.0,
        'sigma_n_N_cm3': 0.4,
        'sigma_phi_V_degrees': 0.8,
        'sigma_ratio': 0.003,
        'sigma_theta_V_degrees': 0.8
    }
    for k, v in fill.items():
        df[k] = df[k].astype(np.float32)
        df[k] = df[k].fillna(v)
    
    df = add_time_features(df)
    df["file_id"] = file_id
    df["total_missing"] = total_missing
    
    df = engineer_space_weather_features(df.tail(OMNI_SEQUENCE_SIZE))
    df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
    for k, v in fill.items():
        df[k] = df[k].astype(np.float32)
        df[k] = df[k].fillna(v)
    
    # Fallback for any new features with NaNs that aren't in the fill dictionary
    # First, check if there are any remaining NaNs
    remaining_nans = df.columns[df.isna().any()].tolist()
    if remaining_nans:
        print("nans remained")
        # For any column with NaNs that isn't in our fill dictionary, use median if available or 0.0 as fallback
        for col in remaining_nans:
            if col not in fill:
                # Try to use median if there are non-NaN values
                median_val = df[col].median()
                if not pd.isna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    # If all values are NaN, use 0.0 as a fallback
                    df[col] = df[col].fillna(0.0)
                # Ensure float32 type consistency
                df[col] = df[col].astype(np.float32)
    return df



def read_process_state_file(fn):
    df = pd.read_csv(fn, parse_dates=["Timestamp"], date_format='%Y-%m-%d %H:%M:%S')
    df = df.set_index(["File ID", "Timestamp"])
    df["has_nan"] = (df>1e5).any(axis=1).astype(int)
    df[df>1e5] = np.nan
    fill = {
        'Semi-major Axis (km)': 6806.180667162488,
        'Eccentricity': 0.0020358706165812747,
        'Inclination (deg)': 88.55308870719918,
        'RAAN (deg)': 179.30555306910912,
        'Argument of Perigee (deg)': 146.8778230277461,
        'True Anomaly (deg)': 205.3289978631664,
        'Latitude (deg)': 17.245090960844102,
        'Longitude (deg)': 0.049281026355355646,
        'Altitude (km)': 447.00621875585676
    }
    for k, v in fill.items():
        df[k] = df[k].fillna(v)
    return df.reset_index().set_index("File ID").sort_index()


def engineer_satellite_features(states_df):
    """
    Create comprehensive physics-informed and data-driven features from satellite orbital elements 
    and position data for atmospheric density prediction and orbital analysis.
    
    Parameters
    ----------
    states_df : DataFrame
        Satellite state data with orbital elements and position information.
        
    Returns
    -------
    DataFrame
        Enhanced states data with engineered features.
    """
    # Create a copy to avoid modifying original data
    df = states_df.copy()
    
    # -------------------------------
    # 0. Basic Setup and Constants
    # -------------------------------
    # Earth parameters
    R_EARTH = 6378.137   # Earth equatorial radius in km (for orbital mechanics)
    R_earth = 6371       # Earth mean radius in km (for altitude computations)
    MU_EARTH = 398600.4418  # Earth's gravitational parameter in km³/s²
    J2 = 1.08262668e-3   # Earth's J2 coefficient (oblateness)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # -------------------------------
    # 1. Angle Conversions & Trig Transformations
    # -------------------------------
    angle_cols = ['Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)',
                  'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)']
    for col in angle_cols:
        # Convert to radians and store as new columns
        df[col + '_rad'] = np.deg2rad(df[col])
        # Create sine and cosine representations for ML
        df[f'{col}_sin'] = np.sin(df[col + '_rad'])
        df[f'{col}_cos'] = np.cos(df[col + '_rad'])
    
    # Compute Argument of Latitude (position in orbital plane relative to ascending node)
    df['Argument of Latitude (deg)'] = (df['Argument of Perigee (deg)'] + df['True Anomaly (deg)']) % 360

    # -------------------------------
    # 2. Basic Orbital Mechanics Features
    # -------------------------------
    # Orbital Period (Kepler's third law)
    df['orbit_period_seconds'] = 2 * np.pi * np.sqrt(df['Semi-major Axis (km)']**3 / MU_EARTH)
    df['orbit_period_minutes'] = df['orbit_period_seconds'] / 60
    # Mean motion (orbits per day)
    df['mean_motion'] = 86400 / df['orbit_period_seconds']
    # Also compute mean motion in rad/s (useful for some calculations)
    df['mean_motion_rad'] = np.sqrt(MU_EARTH / df['Semi-major Axis (km)']**3)

    # Perigee and Apogee Distances and Altitudes
    df['r_perigee'] = df['Semi-major Axis (km)'] * (1 - df['Eccentricity'])
    df['r_apogee'] = df['Semi-major Axis (km)'] * (1 + df['Eccentricity'])
    df['h_perigee'] = df['r_perigee'] - R_earth
    df['h_apogee'] = df['r_apogee'] - R_earth
    # Additional altitude features from second function
    df['perigee_altitude'] = df['h_perigee']  # For naming consistency with second function
    df['apogee_altitude'] = df['h_apogee']    # For naming consistency with second function
    df['altitude_range'] = df['apogee_altitude'] - df['perigee_altitude']
    df['altitude_to_perigee_ratio'] = df['Altitude (km)'] / df['perigee_altitude']
    df['altitude_log'] = np.log10(df['Altitude (km)'])

    # Specific Orbital Energy (km²/s²)
    df['orbital_energy'] = -MU_EARTH / (2 * df['Semi-major Axis (km)'])
    # Specific Angular Momentum (km²/s)
    df['angular_momentum'] = np.sqrt(MU_EARTH * df['Semi-major Axis (km)'] * (1 - df['Eccentricity']**2))
    
    # -------------------------------
    # 3. Orbit Classification & Derived Ratios
    # -------------------------------
    # Classification: Very Low Earth Orbit (VLEO) indicator (perigee altitude below 450 km)
    df['is_VLEO'] = (df['h_perigee'] < 450).astype(int)
    
    # Scale height related: approximate thermospheric scale height ~50 km
    scale_height = 50  # km
    df['scale_height_ratio'] = (df['h_apogee'] - df['h_perigee']) / scale_height
    # Relative density factor assuming an exponential drop with altitude (reference at 400 km)
    h_ref = 400  # km
    df['rel_density_factor'] = np.exp((h_ref - df['h_perigee']) / scale_height)
    
    # -------------------------------
    # 4. Perturbation Influences (J2 Effects)
    # -------------------------------
    df['J2_effect'] = J2 * (R_EARTH / df['Semi-major Axis (km)'])**2
    # Nodal precession rate (degrees per day)
    df['nodal_precession_rate'] = -1.5 * df['J2_effect'] * df['mean_motion'] * np.cos(np.deg2rad(df['Inclination (deg)'])) * (180/np.pi)
    # Perigee precession rate (degrees per day)
    df['perigee_precession_rate'] = 1.5 * df['J2_effect'] * df['mean_motion'] * (2.5 * np.sin(np.deg2rad(df['Inclination (deg)']))**2 - 2) * (180/np.pi)
    
    # -------------------------------
    # 5. Temporal Features & Cyclical Encoding
    # -------------------------------
    # Basic temporal features
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # Fractional hour (from second function)
    df['hour_decimal'] = df['hour'] + df['minute']/60
    
    # Cyclical encoding of time features (from second function)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_decimal'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_decimal'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # -------------------------------
    # 6. Solar/Atmospheric Exposure Features
    # -------------------------------
    # Local solar time approximation (from both functions)
    df['local_hour'] = (df['Longitude (deg)'] / 15.0 + 12) % 24
    # More precise from second function (using decimal hour)
    df['local_hour_precise'] = (df['hour_decimal'] + df['Longitude (deg)']/15) % 24
    df['local_hour_sin'] = np.sin(2 * np.pi * df['local_hour_precise'] / 24)
    df['local_hour_cos'] = np.cos(2 * np.pi * df['local_hour_precise'] / 24)
    
    # Day/night indicator: assume day between 6 and 18 local time
    df['is_dayside'] = ((df['local_hour'] >= 6) & (df['local_hour'] <= 18)).astype(int)
    
    # Solar declination calculation (more accurate formula from second function)
    df['solar_declination'] = 23.45 * np.sin(np.radians(360 * (df['day_of_year'] - 81) / 365))
    
    # Beta angle approximation (angle between orbital plane and sun-earth vector)
    df['beta_angle_approx'] = np.abs(90 - np.abs(df['Inclination (deg)'] - df['solar_declination']))
    
    # Solar hour angle and zenith angle (from second function)
    df['solar_hour_angle'] = (df['local_hour_precise'] - 12) * 15
    
    # Approximate solar zenith angle
    solar_decl_rad = np.radians(df['solar_declination'])
    solar_hour_rad = np.radians(df['solar_hour_angle'])
    lat_rad = np.radians(df['Latitude (deg)'])
    
    cos_zenith = (np.sin(lat_rad) * np.sin(solar_decl_rad) + 
                 np.cos(lat_rad) * np.cos(solar_decl_rad) * np.cos(solar_hour_rad))
    df['solar_zenith_angle'] = np.arccos(np.clip(cos_zenith, -1, 1)) * 180 / np.pi
    
    # Day/night indicator based on solar zenith angle (more accurate than time-based)
    df['is_daytime'] = (df['solar_zenith_angle'] < 90).astype(int)
    
    # -------------------------------
    # 7. Geocentric Cartesian Coordinates
    # -------------------------------
    r = R_earth + df['Altitude (km)']
    lat_rad = np.deg2rad(df['Latitude (deg)'])
    lon_rad = np.deg2rad(df['Longitude (deg)'])
    df['pos_x'] = r * np.cos(lat_rad) * np.cos(lon_rad)
    df['pos_y'] = r * np.cos(lat_rad) * np.sin(lon_rad)
    df['pos_z'] = r * np.sin(lat_rad)
    
    # -------------------------------
    # 8. Orbit Propagation Features
    # -------------------------------
    # True Anomaly in radians (if not already done)
    df['true_anomaly_rad'] = np.deg2rad(df['True Anomaly (deg)'])
    
    # Approximate Eccentric Anomaly using arctan2 formulation
    df['eccentric_anomaly_approx'] = np.arctan2(
        np.sqrt(1 - df['Eccentricity']**2) * np.sin(df['true_anomaly_rad']),
        df['Eccentricity'] + np.cos(df['true_anomaly_rad'])
    )
    # Mean Anomaly (in degrees)
    df['mean_anomaly_deg'] = np.degrees(
        df['eccentric_anomaly_approx'] - df['Eccentricity'] * np.sin(df['eccentric_anomaly_approx'])
    )
    # Fraction of orbit completed (0-1)
    df['orbit_phase'] = (df['mean_anomaly_deg'] % 360) / 360.0
    # Estimate time to perigee (or from perigee) in minutes
    df['minutes_to_perigee'] = df['orbit_period_minutes'] * (1 - df['orbit_phase'])
    df['minutes_from_perigee'] = df['orbit_period_minutes'] * df['orbit_phase']
    
    # Relative velocity at initial position (approximate, km/s)
    df['rel_velocity'] = np.sqrt(
        MU_EARTH * (2 / (R_earth + df['Altitude (km)']) - 1 / df['Semi-major Axis (km)'])
    )
    
    # -------------------------------
    # 9. Normalized or Interaction Features
    # -------------------------------
    # Eccentricity times semi-major axis (capturing orbit "offset")
    df['eccentricity_times_axis'] = df['Eccentricity'] * df['Semi-major Axis (km)']
    
    return df



def engineer_space_weather_features(sw_df):
    """
    Ultimate feature engineering function for OMNI/space weather data.
    Combines physics-informed and statistical features optimized for attention-based models
    with 60-day input windows.
    
    Parameters
    ----------
    sw_df : DataFrame
        Processed space weather data with a 'Timestamp' column and standard OMNI2 variables
        
    Returns
    -------
    DataFrame
        Enhanced space weather data with engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    df = sw_df.copy()
    
    # -------------------------------
    # 1. TIMESTAMP AND BASIC TIME FEATURES
    # -------------------------------
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    
    # Sequence position and time to prediction (for attention models)
    df.reset_index(drop=True, inplace=True)
    df['sequence_position'] = np.arange(len(df)) / len(df)
    df['hours_to_prediction'] = np.arange(-len(df), 0)
    df['normalized_time_to_prediction'] = df['hours_to_prediction'] / len(df)
    
    # -------------------------------
    # 2. SOLAR ACTIVITY FEATURES
    # -------------------------------
    # Rolling averages of f10.7 over 3, 7, and 27 days (only if window fits in 60-day input)
    for window in [3, 7, 27]:
        hours = window * 24
        if hours <= 1440:  # 60 days * 24 = 1440 hours
            df[f'f10.7_avg_{window}d'] = df['f10.7_index'].rolling(window=hours, min_periods=1).mean()
    
    # Daily change rate in F10.7 (difference over 24 hours)
    df['f10.7_change'] = df['f10.7_index'].diff(24).fillna(0)
    
    df["Plasma_Beta"] = np.log1p(df["Plasma_Beta"])
    
    # Solar event flags (using high percentiles of change)
    f10_7_change_threshold = df['f10.7_change'].quantile(0.90)
    f10_7_flare_threshold = df['f10.7_change'].quantile(0.95)
    df['solar_event'] = (df['f10.7_change'] > f10_7_change_threshold).astype(int)
    df['solar_flare'] = (df['f10.7_change'] > f10_7_flare_threshold).astype(int)
    
    # -------------------------------
    # 3. GEOMAGNETIC ACTIVITY FEATURES
    # -------------------------------
    # 24-hour sum of Kp (accumulated geomagnetic activity)
    df['Kp_24h_sum'] = df['Kp_index'].rolling(window=24, min_periods=1).sum()
    
    # Maximum ap index in last 12 hours
    df['ap_12h_max'] = df['ap_index_nT'].rolling(window=12, min_periods=1).max()
    
    # 6-hour difference in Dst index (indicating storm development)
    df['Dst_change_6h'] = df['Dst_index_nT'].diff(6).fillna(0)
    
    # Combined geomagnetic activity index (normalized combination)
    df['geo_index'] = (df['Kp_index'] / 9.0) + (np.abs(df['Dst_index_nT']) / 100) + (df['ap_index_nT'] / 400)
    
    # Geomagnetic storm flag (if any of the criteria is met)
    df['geomagnetic_storm'] = ((df['Kp_index'] >= 5) | 
                               (df['Dst_index_nT'] <= -50) |
                               (df['ap_index_nT'] >= 50)).astype(int)
    
    # Time since most recent storm event (cumulative count reset at each storm)
    storm_groups = df['geomagnetic_storm'].cumsum()
    df['hours_since_storm'] = storm_groups.groupby(storm_groups).cumcount()
    
    # -------------------------------
    # 4. MAGNETIC FIELD FEATURES
    # -------------------------------
    # Ratio of vector to scalar magnetic field magnitude
    df['B_vec_to_scalar_ratio'] = df['Vector_B_Magnitude_nT'] / df['Scalar_B_nT']
    
    # Trigonometric representations for GSE magnetic field direction angles
    for angle in ['Lat_Angle_of_B_GSE', 'Long_Angle_of_B_GSE']:
        df[angle + '_rad'] = np.deg2rad(df[angle])
        df[angle + '_sin'] = np.sin(df[angle + '_rad'])
        df[angle + '_cos'] = np.cos(df[angle + '_rad'])
    
    # Differences between components in GSE and GSM coordinates
    df['B_GSE_diff_BY_BZ'] = df['BY_nT_GSE'] - df['BZ_nT_GSE']
    df['B_GSM_diff_BY_BZ'] = df['BY_nT_GSM'] - df['BZ_nT_GSM']
    
    # RMS ratio: RMS field vector divided by RMS magnitude
    df['RMS_ratio_field_to_mag'] = df['RMS_field_vector_nT'] / (df['RMS_magnitude_nT'] + 1e-8)
    
    # -------------------------------
    # 5. SOLAR WIND & IMF FEATURES
    # -------------------------------
    # Plasma flow angles: trigonometric features for both longitude and latitude
    for flow_angle in ['SW_Plasma_flow_long_angle', 'SW_Plasma_flow_lat_angle']:
        df[flow_angle + '_rad'] = np.deg2rad(df[flow_angle])
        df[flow_angle + '_sin'] = np.sin(df[flow_angle + '_rad'])
        df[flow_angle + '_cos'] = np.cos(df[flow_angle + '_rad'])
    
    # Southward IMF Bz component: critical for geomagnetic coupling
    df['BZ_southward'] = df['BZ_nT_GSM'].clip(upper=0).abs()
    
    # Rectified electric field: combining solar wind speed and southward IMF Bz
    df['rectified_E_field'] = df['SW_Plasma_Speed_km_s'] * df['BZ_southward'] / 1000.0
    
    # Solar wind ram pressure 
    df['SW_ram_pressure'] = df['SW_Proton_Density_N_cm3'] * (df['SW_Plasma_Speed_km_s'] ** 2) / 1.0e6
    
    # Magnetosonic Mach number multiplied by southward IMF
    if 'Magnetosonic_Mach_number' in df.columns:
        df['MS_Mach_South'] = df['Magnetosonic_Mach_number'] * df['BZ_southward']
    
    # Clock angle parameters: convert IMF BY and BZ to sine and cosine of clock angle
    denominator = np.sqrt(df['BY_nT_GSM']**2 + df['BZ_nT_GSM']**2)
    df['IMF_clock_sin'] = np.where(denominator > 0, df['BY_nT_GSM'] / denominator, 0)
    df['IMF_clock_cos'] = np.where(denominator > 0, df['BZ_nT_GSM'] / denominator, 0)
    
    # Newell coupling function: physics-based solar wind-magnetosphere coupling
    df['newell_coupling'] = (df['SW_Plasma_Speed_km_s'] ** (4/3)) * \
                            (df['Vector_B_Magnitude_nT'] ** (2/3)) * \
                            (np.sin(np.arctan2(df['BY_nT_GSM'], df['BZ_nT_GSM']) / 2))
    
    # Energy coupling function: energy transfer from solar wind to magnetosphere
    df['energy_coupling'] = (df['SW_Plasma_Speed_km_s'] ** (4/3)) * \
                           (df['Vector_B_Magnitude_nT'] ** (2/3)) * \
                           (df['BZ_southward'] ** (8/3))
    
    # Calculate approximate magnetopause standoff distance
    df['magnetopause_standoff'] = (df['SW_ram_pressure'] + 1e-10) ** (-1/6)
    
    # -------------------------------
    # 6. VARIABILITY & QUALITY METRICS
    # -------------------------------
    # Average number of data points (IMF and Plasma) as a quality measure
    df['avg_points'] = (df['num_points_IMF_averages'] + df['num_points_Plasma_averages']) / 2.0
    
    # Ratios of standard deviations to their corresponding measurements
    df['temp_sigma_ratio'] = df['sigma_T_K'] / (df['SW_Plasma_Temperature_K'] + 1e-8)
    df['density_sigma_ratio'] = df['sigma_n_N_cm3'] / (df['SW_Proton_Density_N_cm3'] + 1e-8)
    df['speed_sigma_ratio'] = df['sigma_V_km_s'] / (df['SW_Plasma_Speed_km_s'] + 1e-8)
    
    # -------------------------------
    # 7. PARTICLE FLUX FEATURES
    # -------------------------------
    # For columns with moderate missingness (~20%), log-transform after filling NaNs with a small value
    for col in ['Proton_flux_>10_Mev', 'Proton_flux_>30_Mev', 'Proton_flux_>60_Mev']:
        if col in df.columns:
            df[col + '_log'] = np.log(df[col].fillna(1e-8) + 1e-8)
    
    # Drop particle flux columns with extremely high missingness (~89%)
    cols_to_drop = ['Proton_flux_>1_Mev', 'Proton_flux_>2_Mev', 'Proton_flux_>4_Mev']
    df = df.drop(columns=cols_to_drop)
    
    return df
