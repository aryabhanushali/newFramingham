import pandas as pd
import numpy as np

def build_activity_features(paxday, paxhd):
    """
    Build per-participant wearable features from daily activity data.
    These are the ONLY features your model will use — no blood labs.
    """

    # Merge with header to get valid wear flags
    df = paxday.merge(paxhd[['SEQN', 'PAXHAND']], on='SEQN', how='left')

    # Filter to valid days only (PAXDAYWEAR >= 600 mins = 10 hours)
    if 'PAXDAYWEAR' in df.columns:
        df = df[df['PAXDAYWEAR'] >= 600]

    features = []

    for seqn, person in df.groupby('SEQN'):
        row = {'SEQN': seqn}

        # --- Step count features ---
        if 'PAXDAYSTEP' in person.columns:
            steps = person['PAXDAYSTEP'].dropna()
            row['mean_daily_steps']    = steps.mean()
            row['std_daily_steps']     = steps.std()
            row['min_daily_steps']     = steps.min()
            row['max_daily_steps']     = steps.max()
            row['days_under_5k_steps'] = (steps < 5000).sum()
            row['days_over_10k_steps'] = (steps > 10000).sum()

        # --- Wear time features ---
        if 'PAXDAYWEAR' in person.columns:
            wear = person['PAXDAYWEAR'].dropna()
            row['mean_wear_minutes'] = wear.mean()
            row['n_valid_days']      = len(wear)

        # --- Sedentary time ---
        if 'PAXDAYSED' in person.columns:
            sed = person['PAXDAYSED'].dropna()
            row['mean_sedentary_mins'] = sed.mean()
            row['sedentary_ratio']     = (sed / person['PAXDAYWEAR']).mean()

        # --- Moderate-vigorous activity ---
        if 'PAXDAYMVPA' in person.columns:
            mvpa = person['PAXDAYMVPA'].dropna()
            row['mean_mvpa_mins']       = mvpa.mean()
            row['days_meet_guidelines'] = (mvpa >= 30).sum()

        # --- Activity regularity (lower std = more regular) ---
        if 'PAXDAYSTEP' in person.columns:
            row['activity_regularity'] = 1 / (steps.std() + 1)

        features.append(row)

    return pd.DataFrame(features)


def build_model_dataset(clinical_scored, paxday, paxhd):
    """
    Merge Framingham labels with wearable-only features.
    Drop all clinical lab values — model can only see wearable data.
    """
    # Build activity features
    act = build_activity_features(paxday, paxhd)

    # Keep only wearable-safe columns from clinical
    wearable_safe = ['SEQN', 'RIDAGEYR', 'RIAGENDR',
                     'framingham_risk', 'high_risk']
    clinical_clean = clinical_scored[wearable_safe].dropna(subset=['framingham_risk'])

    # Merge
    dataset = clinical_clean.merge(act, on='SEQN', how='inner')

    print(f"Final dataset: {dataset.shape[0]} participants, {dataset.shape[1]} features")
    print(f"High risk: {dataset['high_risk'].sum()} ({dataset['high_risk'].mean()*100:.1f}%)")
    print(f"Low risk:  {(dataset['high_risk']==0).sum()}")
    print(f"\nFeature columns:\n{[c for c in dataset.columns if c not in wearable_safe]}")

    return dataset


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.load_data import load_all
    from src.framingham import compute_framingham

    clinical, paxhd, paxday, paxhr = load_all()
    clinical_scored = compute_framingham(clinical)
    dataset = build_model_dataset(clinical_scored, paxday, paxhd)
    print(dataset.head())