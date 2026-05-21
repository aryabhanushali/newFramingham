import pandas as pd
import numpy as np

# PAXDAY column guide (NHANES 2011-2012):
# PAXVMD   = valid wear minutes per day
# PAXWWMD  = wake wear minutes
# PAXSWMD  = sleep wear minutes
# PAXNWMD  = non-wear minutes
# PAXAISMD = total activity intensity sum (accelerometer counts)
# PAXMTSD  = mean activity intensity
# PAXLXSD  = ambient light sum (lux)

def build_activity_features(paxday, paxhd):

    df = paxday.merge(paxhd[['SEQN', 'PAXHAND']], on='SEQN', how='left')

    # Filter to valid days only — at least 600 valid wear minutes
    df = df[df['PAXVMD'] >= 600].copy()

    features = []

    for seqn, person in df.groupby('SEQN'):
        row = {'SEQN': seqn}

        # --- Activity intensity (main signal) ---
        activity = person['PAXAISMD'].dropna()
        row['mean_activity']       = activity.mean()
        row['std_activity']        = activity.std()
        row['min_activity']        = activity.min()
        row['max_activity']        = activity.max()
        row['activity_regularity'] = 1 / (activity.std() + 1)

        # Low activity days (bottom quartile of this person's data)
        threshold = activity.quantile(0.25)
        row['low_activity_days']  = (activity < threshold).sum()
        row['high_activity_days'] = (activity > activity.quantile(0.75)).sum()

        # --- Mean activity intensity per minute (normalised) ---
        mean_intensity = person['PAXMTSD'].dropna()
        row['mean_intensity']     = mean_intensity.mean()
        row['std_intensity']      = mean_intensity.std()

        # --- Wake vs sleep time ---
        wake  = person['PAXWWMD'].dropna()
        sleep = person['PAXSWMD'].dropna()
        row['mean_wake_mins']     = wake.mean()
        row['mean_sleep_mins']    = sleep.mean()
        row['sleep_ratio']        = (sleep / (wake + sleep + 1)).mean()

        # --- Non-wear / sedentary proxy ---
        nonwear = person['PAXNWMD'].dropna()
        row['mean_nonwear_mins']  = nonwear.mean()

        # --- Light exposure (circadian proxy) ---
        lux = person['PAXLXSD'].dropna()
        row['mean_lux']           = lux.mean()
        row['std_lux']            = lux.std()

        # --- Valid days count ---
        row['n_valid_days']       = len(person)

        features.append(row)

    return pd.DataFrame(features)


def build_model_dataset(clinical_scored, paxday, paxhd):

    act = build_activity_features(paxday, paxhd)

    wearable_safe = ['SEQN', 'RIDAGEYR', 'RIAGENDR',
                     'framingham_risk', 'high_risk']
    clinical_clean = clinical_scored[wearable_safe].dropna(subset=['framingham_risk'])

    dataset = clinical_clean.merge(act, on='SEQN', how='inner')

    feature_cols = [c for c in dataset.columns if c not in wearable_safe]

    print(f"Final dataset: {dataset.shape[0]} participants, {len(feature_cols)} features")
    print(f"High risk: {dataset['high_risk'].sum()} ({dataset['high_risk'].mean()*100:.1f}%)")
    print(f"Low risk:  {(dataset['high_risk']==0).sum()}")
    print(f"\nFeatures: {feature_cols}")

    return dataset