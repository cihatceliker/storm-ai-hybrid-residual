from util import *
from orbit import *

# paths
TEST_DATA_DIR = Path("data")
INITIAL_STATES_PATH = os.path.join(TEST_DATA_DIR, 'initial_states.csv')
omni2_path = os.path.join(TEST_DATA_DIR, "omni2")

# read and prepare initial states
initial_states = read_process_state_file(INITIAL_STATES_PATH)
initial_states = engineer_satellite_features(initial_states)
initial_states["omni_path"] = "omni2-" + ("00000"+initial_states.index.astype(str)).str.slice(-5) + ".csv"

DEBUG = 0
if DEBUG:
    initial_states = initial_states.tail(5)

# read and prepare omni files
omni_df = pd.concat([read_process_omni_file(os.path.join(omni2_path, str(row["omni_path"])), 
	                                    file_id=int(i)) 
	             for i, row in tqdm(initial_states.iterrows(), total=len(initial_states))], ignore_index=True)

# making the distributions better.
# this file has min values of the log transformed features
# calculated from training. same used on inference.
min_vals = pd.read_csv("assets/min_vals.csv", index_col=0)["0"].to_dict()
ls = []
for col in OMNI_LOGGED:
    # shift to positive
    a = omni_df[col] - min_vals[col] + 1
    # for potential outliers in inference
    a = a.clip(lower=1)
    # log transform
    ls.append(np.log(a).rename(col+"_logged"))
omni_df = pd.concat([omni_df, *ls], axis=1)

# add static features from omni data
initial_states, omni_df = statics_from_omni(initial_states, omni_df)
omni_df = omni_df.sort_values(["Timestamp", "file_id"], ignore_index=True)

# physics features(orbit, msis etc) followed by a little feat eng.
physics = create_physics_features(initial_states, omni_df)
physics_feats = physics.groupby("file_id").apply(feat_eng_physics)
initial_states = pd.concat([initial_states, physics_feats], axis=1)

initial_states = initial_states.reset_index().rename(columns={"index": "File ID"})

if not DEBUG:
    initial_states.to_feather(f"cache/{TEST_DATA_DIR}_initial_states.ft")
    physics.to_feather(f"cache/{TEST_DATA_DIR}_physics.ft")
    omni_df.to_feather(f"cache/{TEST_DATA_DIR}_omni_df.ft")

# now prepare target data
df = pd.concat([pd.read_csv(f"{TEST_DATA_DIR}/sat_density/"+f).assign(file_id=int(f[7:12]), category=f[:6]) for f in 
	        tqdm(os.listdir(f"{TEST_DATA_DIR}/sat_density/"))])
# extremely high values are nans
df.loc[df["Orbit Mean Density (kg/m^3)"]>1e-5, "Orbit Mean Density (kg/m^3)"] = np.nan
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# now we prepare target df with correct format
start = df.groupby("file_id")["Timestamp"].min()
s = df.file_id.value_counts()
for id_ in tqdm(s[s<432].index):
    new_rows = 432-s.loc[id_]
    row = df[df["file_id"]==id_].iloc[0]
    df = pd.concat([df, pd.DataFrame({
	"Timestamp": [start[id_]+timedelta(minutes=10 * f) for f in range(new_rows)],
	"file_id": [id_ for f in range(new_rows)],
	"category": [row["category"] for f in range(new_rows)],
    })], ignore_index=True)
print(s[s<432].index.tolist())

# bad flag. will be used on metric and also when calculating loss
df["bad_flag"] = df["Orbit Mean Density (kg/m^3)"].isna()
df = df.groupby("file_id").head(432)

# if all target is nan drop the satellite
s = df.groupby("file_id").bad_flag.mean()
df = df[~df["file_id"].isin(s[s==1].index)].copy()

df.sort_values(["file_id", "Timestamp"], ignore_index=True, inplace=True)
df.to_feather(f"cache/{TEST_DATA_DIR}_sat_density.ft")
