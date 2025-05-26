from util import *


def train_with_val(train_dataset, val_dataset, epochs, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = SatelliteDensityModelV4(len(OMNI_FEATURES), len(STATE_FEATURES), n_phys=len(DENSITY_FEATURES)).to(DEVICE)
    criterion = DirectWeightedRMSELoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_epoch = -1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for omni, static, targets, densities, bad_flags, sat_ids in train_loader:
            omni = omni.to(DEVICE)
            static = static.to(DEVICE)
            targets = targets.to(DEVICE)
            bad_flags = bad_flags.to(DEVICE)
            densities = [densities[:,:,i].to(DEVICE) for i in range(len(DENSITY_FEATURES))]
            
            predictions = model(omni, static, densities)
            loss = criterion(predictions, targets, bad_flags)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for omni, static, targets, densities, bad_flags, sat_id in val_loader:
                omni = omni.to(DEVICE)
                static = static.to(DEVICE)
                targets = targets.to(DEVICE)
                bad_flags = bad_flags.to(DEVICE)
                densities = [densities[:,:,i].to(DEVICE) for i in range(len(DENSITY_FEATURES))]
                predictions = model(omni, static, densities)
                loss = criterion(predictions, targets, bad_flags)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        scheduler.step(val_loss)
        
        #save if 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'assets/model.pt')
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best Val Loss: {best_val_loss:.6f}")
    print()
    print("BEST EPOCH", best_epoch)
    model.load_state_dict(torch.load(f'assets/model.pt', weights_only=True))
    return model, train_losses, val_losses


def eval_score(model, val_loader, batch_size):
    results = {}
    model.eval()
    with torch.no_grad():
        for omni, static, targets, densities, bad_flags, sat_id in val_loader:
            omni = omni.to(DEVICE)
            static = static.to(DEVICE)
            targets = targets.to(DEVICE)
            bad_flags = bad_flags.to(DEVICE)
            densities = [densities[:,:,i].to(DEVICE) for i in range(len(DENSITY_FEATURES))]
            
            predictions = model(omni, static, densities)
            # getting predictions on the right scale depending on which target used
            if USE_RATIO:
                predictions = torch.exp(
                    densities[1] * 
                    physics_scaler.scale_[1] + 
                    physics_scaler.mean_[1]
                ) / torch.exp(predictions)
            else:
                predictions = (predictions * STD) + torch.exp(
                    densities[1] * 
                    physics_scaler.scale_[1] + 
                    physics_scaler.mean_[1]
                )
            
            results.update(dict(zip(list(sat_id.detach().cpu().numpy()), [f for f in predictions.cpu()])))
    
    cat_map = initial_states[initial_states["File ID"].isin(results.keys())].set_index("File ID")["category"]
    scores = {}
    rmse_scores, skill_scores, msis_scores = {}, {}, {}
    for k in results.keys():
        a = target_df[target_df["file_id"]==k]["Orbit Mean Density (kg/m^3)"]
        b = results[k].cpu().numpy()
        c = target_df[target_df["file_id"]==k]["msis"]
        rmse = metric(a, b, target_df[target_df["file_id"]==k]["bad_flag"])
        baseline = metric(c, b, target_df[target_df["file_id"]==k]["bad_flag"])
        score = 1 - (rmse / baseline)
        scores[k] = score
        skill_scores[k] = [cat_map[k], score]
        rmse_scores[k] = [cat_map[k], rmse]
        msis_scores[k] = [cat_map[k], baseline]
    score = np.mean(list(scores.values()))
    scores_dict = {
        "total": score, 
        "skill scores": pd.DataFrame(skill_scores).T.groupby(0)[1].mean().to_dict(), 
        "rmse scores": pd.DataFrame(rmse_scores).T.groupby(0)[1].mean().to_dict(), 
        "msis scores": pd.DataFrame(msis_scores).T.groupby(0)[1].mean().to_dict(), 
    }
    return results, scores_dict


# couple parameters
BATCH_SIZE = 64
DEBUG = 0
USE_RATIO = 0#target, log diff or log ratio
TARGET_COL = "Orbit Mean Density (kg/m^3)"


if __name__=="__main__":
    # read target data
    target_df = pd.read_feather("cache/data_sat_density.ft")
    print(target_df.file_id.nunique())
    complete_ones = target_df["file_id"].unique()

    if DEBUG:
        complete_ones = np.random.choice(complete_ones, 200)
        target_df = target_df[target_df["file_id"].isin(complete_ones)].reset_index(drop=True)

    # dropping the ones that have high number of nans in target values
    s = target_df.groupby("file_id")["bad_flag"].mean()
    a = s[s>=0.8].index
    target_df = target_df[~target_df["file_id"].isin(a)].reset_index(drop=True)
    complete_ones = target_df["file_id"].unique()

    # space weather data
    omni_df = pd.read_feather("cache/data_omni_df.ft")
    omni_df = omni_df[omni_df["file_id"].isin(complete_ones)]
    omni_df.reset_index(drop=True, inplace=True)

    # state data
    initial_states = pd.read_feather("cache/data_initial_states.ft")
    del initial_states["omni_path"]
    initial_states = initial_states[initial_states["File ID"].isin(complete_ones)].reset_index(drop=True)
    initial_states["category"] = initial_states["File ID"].map(target_df.groupby("file_id")["category"].first())

    # physics data
    physics = pd.read_feather("cache/data_physics.ft")
    physics = physics[physics["file_id"].isin(complete_ones)]
    physics["category"] = physics["file_id"].map(initial_states.set_index("File ID")["category"])

    # adding msis values from physics data to target df
    assert physics.shape[0] == target_df.shape[0]
    physics.sort_values(["file_id","Timestamp"], ignore_index=True, inplace=True)
    target_df.sort_values(["file_id","Timestamp"], ignore_index=True, inplace=True)
    omni_df.sort_values(["file_id", "Timestamp"], ignore_index=True, inplace=True)
    # exp for original scale
    target_df["msis"] = np.exp(physics["msis"])
    target_df["dtm"] = np.exp(physics["dtm"])

    # creating targets
    target_df["interpolated_ratio"] = np.log(target_df["msis"] / target_df[TARGET_COL])
    target_df["interpolated_residual"] = (target_df[TARGET_COL] - target_df["msis"])
    STD = target_df["interpolated_residual"].std()
    # scaling log diff target. log ratio is already on a good scale
    target_df["interpolated_residual"] /= STD
    print(STD)

    # dropping the ones that have nans values in initialstates data
    a = initial_states[initial_states["has_nan"]==0]["File ID"].values
    initial_states = initial_states[initial_states["File ID"].isin(a)].reset_index(drop=True)
    target_df = target_df[target_df["file_id"].isin(a)].reset_index(drop=True)
    physics = physics[physics["file_id"].isin(a)].reset_index(drop=True)
    omni_df = omni_df[omni_df["file_id"].isin(a)].reset_index(drop=True)
    if "has_nan" in initial_states.columns:
        del initial_states["has_nan"]

    # creating features per dataframe
    OMNI_FEATURES = sorted(omni_df.columns.difference(['Timestamp', 'file_id']))
    DENSITY_FEATURES = sorted(physics.columns.difference(['Timestamp', 'category', 'dtm', 'file_id']))
    STATE_FEATURES = sorted(initial_states.columns.difference(['File ID', 'Timestamp', 'category']))
    joblib.dump([STATE_FEATURES, OMNI_FEATURES, DENSITY_FEATURES], "assets/features.jb")

    # standard scaling all features
    scale_these = [f for f in STATE_FEATURES if not ("sin" in f or "cos" in f or "sample_weight" in f or f.startswith("is_"))]
    state_scaler = StandardScaler()
    omni_scaler = StandardScaler()
    physics_scaler = StandardScaler()
    initial_states[scale_these] = state_scaler.fit_transform(initial_states[scale_these])
    omni_df[OMNI_FEATURES] = omni_scaler.fit_transform(omni_df[OMNI_FEATURES])
    physics[DENSITY_FEATURES] = physics_scaler.fit_transform(physics[DENSITY_FEATURES])
    # saving for inference
    joblib.dump([state_scaler, omni_scaler, physics_scaler, STD], "assets/scalers.jb")

    # adding folds
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for fold, (tr, vl) in enumerate(skf.split(initial_states["File ID"], 
                                              y=initial_states["category"])):
        val_ids = initial_states["File ID"].iloc[vl].sort_values().values
        initial_states.loc[initial_states["File ID"].isin(val_ids), "fold"] = fold
    initial_states["fold"] = initial_states["fold"].astype(int)

    # which target to use
    if USE_RATIO:
        use_target = target_df["interpolated_ratio"].name
    else:
        use_target = target_df["interpolated_residual"].name
    print(STD, USE_RATIO)
    
    all_train_losses = []
    all_val_losses = []
    for fold in sorted(initial_states["fold"].unique()):
        train_ids = initial_states[initial_states["fold"]!=fold]["File ID"].sort_values().values
        val_ids = initial_states[initial_states["fold"]==fold]["File ID"].sort_values().values
        print("fold", fold)

        train_dataset = SatelliteDataset(omni_df[omni_df["file_id"].isin(train_ids)], 
                                         initial_states[initial_states["File ID"].isin(train_ids)], 
                                         target_df[target_df["file_id"].isin(train_ids)], 
                                         physics[physics["file_id"].isin(train_ids)], 
                                         target_col_name=use_target,
                                         omni_features=OMNI_FEATURES, 
                                         state_features=STATE_FEATURES, 
                                         physics_features=DENSITY_FEATURES)

        val_dataset = SatelliteDataset(omni_df[omni_df["file_id"].isin(val_ids)], 
                                       initial_states[initial_states["File ID"].isin(val_ids)], 
                                       target_df[target_df["file_id"].isin(val_ids)], 
                                       physics[physics["file_id"].isin(val_ids)], 
                                       target_col_name=use_target,
                                       omni_features=OMNI_FEATURES, 
                                       state_features=STATE_FEATURES, 
                                       physics_features=DENSITY_FEATURES)
        best_model, train_losses, val_losses = train_with_val(train_dataset, val_dataset, 
                                                              epochs=40 if not DEBUG else 3, 
                                                              batch_size=BATCH_SIZE)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        if not DEBUG:
            torch.save(best_model.state_dict(), f'assets/model_fold_{fold}.pt')
    
    print("MODELS SAVED")
    
    oof_preds_d = {}
    fold_scores = []
    for fold in sorted(initial_states["fold"].unique()):
        val_ids = initial_states[initial_states["fold"]==fold]["File ID"].sort_values().values
        val_dataset = SatelliteDataset(omni_df[omni_df["file_id"].isin(val_ids)], 
                                       initial_states[initial_states["File ID"].isin(val_ids)], 
                                       target_df[target_df["file_id"].isin(val_ids)], 
                                       physics[physics["file_id"].isin(val_ids)], 
                                       target_col_name=use_target,
                                       omni_features=OMNI_FEATURES, 
                                       state_features=STATE_FEATURES, 
                                       physics_features=DENSITY_FEATURES)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        model = SatelliteDensityModelV4(len(OMNI_FEATURES), len(STATE_FEATURES), 
                                        n_phys=len(DENSITY_FEATURES)).to(DEVICE)
        model.load_state_dict(torch.load(f'assets/model_fold_{fold}.pt', weights_only=True))
        results, scores_dict = eval_score(model, val_loader, batch_size=BATCH_SIZE)
        fold_scores.append(scores_dict)
        oof_preds_d.update(results)
    
    # preparing oof results
    oof_preds = pd.DataFrame(oof_preds_d).melt().rename(columns={"variable": "file_id", "value": "prediction"})
    oof_preds["step"] = oof_preds.index%432
    target_df = target_df.sort_values(["file_id", "Timestamp"], ignore_index=True)
    oof_preds = oof_preds.sort_values(["file_id", "step"], ignore_index=True)
    target_df["step"] = target_df.index%432
    oof_preds[TARGET_COL] = target_df[TARGET_COL]
    oof_preds["msis"] = target_df["msis"]
    oof_preds["category"] = target_df["category"]
    oof_preds["bad_flag"] = target_df["bad_flag"]
    
    oof_rmse_total = oof_preds.groupby("file_id").apply(lambda x: metric(x["prediction"], x[TARGET_COL], x["bad_flag"]))
    oof_msis_total = oof_preds.groupby("file_id").apply(lambda x: metric(x["msis"], x[TARGET_COL], x["bad_flag"]))
    oof_skill_total = 1 - oof_rmse_total / oof_msis_total

    oof_scores = pd.concat([oof_rmse_total.rename("rmse"), 
                            oof_msis_total.rename("msis"), 
                            oof_skill_total.rename("skill")], axis=1)
    oof_scores["category"] = initial_states.set_index("File ID")["category"]
    print("OOF RESULTS")
    print(oof_scores.groupby("category").mean().to_dict())
    print(oof_scores[["rmse","msis","skill"]].mean())
