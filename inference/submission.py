from util import *
from copy import deepcopy
from orbit import *


TRAINED_MODEL_PATH = Path('assets/model.pt')
FEATURES_PATH = Path('assets/features.jb')
SCALERS_PATH = Path('assets/scalers.jb')
OMNI_SHIFT_VALS_PATH = Path('assets/omni_min_vals.jb')
TEST_DATA_DIR = os.path.join('/app','data', 'dataset', 'test')
TEST_PREDS_FP = Path('/app/output/prediction.json')
INITIAL_STATES_PATH = Path('/app/input_data/initial_states.csv')
omni2_path = os.path.join(TEST_DATA_DIR, "omni2")

DEBUG = 0
if DEBUG:
    DEVICE = torch.device("cpu")
    TEST_DATA_DIR = Path("temp")
    TEST_PREDS_FP = Path('temp/prediction.json')
    INITIAL_STATES_PATH = Path('temp/initial_states.csv')
    omni2_path = os.path.join(TEST_DATA_DIR, "omni2")


initial_states = read_process_state_file(INITIAL_STATES_PATH)
initial_states = engineer_satellite_features(initial_states)
initial_states["omni_path"] = "omni2-" + ("00000"+initial_states.index.astype(str)).str.slice(-5) + ".csv"

omni_df = pd.concat([read_process_omni_file(os.path.join(omni2_path, str(row["omni_path"])), 
                                            file_id=int(i)) 
                     for i, row in initial_states.iterrows()], ignore_index=True)

# making the distributions better
min_vals = {'BZ_nT_GSE': -53.70000076293945,'BZ_nT_GSM': -57.79999923706055,'Dst_index_nT': -422.0,'pc_index': -21.5,'sigma_theta_V_degrees': 0.0,'Alfen_mach_number': 0.6000000238418579,'f10.7_change': -135.10000610351562,'hours_since_storm': 0.0,'ap_index_nT': 0.0,'Alpha_Prot_ratio': 0.0010000000474974513,'RMS_field_vector_nT': 0.0,'MS_Mach_South': 0.0,'SW_Proton_Density_N_cm3': 0.10000000149011612,'RMS_BX_GSE_nT': 0.0,'sigma_V_km_s': 0.0,'RMS_ratio_field_to_mag': 0.0,'speed_sigma_ratio': 0.0,'BZ_southward': 0.0,'RMS_BY_GSE_nT': 0.0,'Dst_change_6h': -392.0,'E_electric_field': -35.939998626708984,'RMS_BZ_GSE_nT': 0.0,'sigma_phi_V_degrees': 0.0,'temp_sigma_ratio': 0.0,'Flow_pressure': 0.029999999329447746,'SW_ram_pressure': 0.014212900772690773,'rectified_E_field': 0.0,'RMS_magnitude_nT': 0.0,'SW_Plasma_flow_lat_angle_cos': 0.9128341674804688,'sigma_n_N_cm3': 0.0,'SW_Plasma_Temperature_K': 3299.0,'SW_Plasma_flow_long_angle_cos': 0.8309844732284546,'Quasy_Invariant': 9.999999747378752e-05,'sigma_ratio': 0.0,'Proton_flux_>10_Mev': 0.009999999776482582,'Proton_flux_>30_Mev': 0.009999999776482582,'Proton_flux_>60_Mev': 0.009999999776482582,'sigma_T_K': 0.0,'energy_coupling': 0.0,'Kp_index': 0.0,'f10.7_index': 63.400001525878906,'AE_index_nT': 3.0,'AL_index_nT': -2452.0,'AU_index_nT': -260.0}

# making the distributions better
ls = []
for col in OMNI_LOGGED:
    # shift to positive
    a = omni_df[col] - min_vals[col] + 1
    # for potential outliers in inference
    a = a.clip(lower=1)
    # log transform
    ls.append(np.log(a).rename(col+"_logged"))
omni_df = pd.concat([omni_df, *ls], axis=1)


initial_states, omni_df = statics_from_omni(initial_states, omni_df)
omni_df = omni_df.sort_values(["Timestamp", "file_id"], ignore_index=True)

if DEBUG:
    initial_states.loc[initial_states.index[0], "Timestamp"] = np.nan
    omni_df.drop(omni_df[omni_df["file_id"]==3703].index[:-2], inplace=True)

physics = create_physics_features(initial_states, omni_df)

broken_msis = set(omni_df["file_id"].unique()).difference(physics["file_id"])
if len(broken_msis) > 0:
    print("filling msis", broken_msis)
    for id_ in broken_msis:
        ff = physics[physics["file_id"]==physics["file_id"].iloc[0]].copy()
        ff["file_id"] = id_
        physics = pd.concat([physics, ff], ignore_index=True)
    del ff


physics_feats = physics.groupby("file_id").apply(feat_eng_physics)
initial_states = pd.concat([initial_states, physics_feats], axis=1)

###################
###PREP_DATA_END###
###################

state_scaler, omni_scaler, physics_scaler, STD = joblib.load(SCALERS_PATH)
STATE_FEATURES, OMNI_FEATURES, DENSITY_FEATURES = joblib.load(FEATURES_PATH)
input_dim = len(OMNI_FEATURES)
static_dim = len(STATE_FEATURES)
physics[DENSITY_FEATURES] = physics_scaler.transform(physics[DENSITY_FEATURES])
omni_df[OMNI_FEATURES] = omni_scaler.transform(omni_df[OMNI_FEATURES])
scale_these = [f for f in STATE_FEATURES if not ("sin" in f or "cos" in f or f.startswith("is_"))]
initial_states[scale_these] = state_scaler.transform(initial_states[scale_these])

initial_states.index.name = "File ID"
initial_states = initial_states.copy()
initial_states.reset_index(inplace=True)
omni_df.sort_values(["file_id", "Timestamp"], ignore_index=True, inplace=True)

names = [f for f in os.listdir("assets") if "model_fold" in f]
models = []
for name in names:
    model = SatelliteDensityModelV4(input_dim, static_dim, n_phys=len(DENSITY_FEATURES)).to(DEVICE).eval()
    model.load_state_dict(torch.load(os.path.join("assets", name), map_location=torch.device("cpu"), weights_only=True))
    models.append(deepcopy(model))
#model = SatelliteDensityModelV4(148, 229, n_phys=4).to(DEVICE).eval()
#model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))


dataset = SatelliteDatasetEval(omni_df, initial_states, physics, 
                               OMNI_FEATURES, STATE_FEATURES, DENSITY_FEATURES)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

all_preds = {}
final_predictions = {}
model.eval()
with torch.no_grad():
    for omni, static, densities, sat_id in test_loader:
        omni = omni.to(DEVICE)
        static = static.to(DEVICE)
        densities = [densities[:,:,i].to(DEVICE) for i in range(len(DENSITY_FEATURES))]
        
        preds = []
        for model in models:
            model.eval()
            predictions = model(omni, static, densities)
            preds.append(predictions)
        preds = torch.stack(preds).mean(dim=0)
        #preds = torch.exp(densities[1]*physics_scaler.scale_[1]+physics_scaler.mean_[1]) / torch.exp(preds)
        preds = (preds * STD) + torch.exp(densities[1]*physics_scaler.scale_[1]+physics_scaler.mean_[1])
        all_preds.update(dict(zip(sat_id.detach().cpu().numpy(), 
                                  [f.detach().cpu().numpy() for f in preds])))

for id_, preds in all_preds.items():
    start = initial_states.set_index("File ID")["Timestamp"].loc[id_]
    timestamps = [start+timedelta(minutes=10 * i) for i in range(432)]
    final_predictions[str(id_)] = {
        "Timestamp": list(map(lambda ts: ts.isoformat(), timestamps)), 
        "Orbit Mean Density (kg/m^3)": list(map(float, preds))
    }

with open(TEST_PREDS_FP, "w") as outfile: 
    json.dump(final_predictions, outfile)
print("Saved predictions to: {}".format(TEST_PREDS_FP))
predictions = final_predictions
if DEBUG:
    print(predictions["8116"]["Orbit Mean Density (kg/m^3)"][:5], predictions["8116"]["Orbit Mean Density (kg/m^3)"][-5:])
