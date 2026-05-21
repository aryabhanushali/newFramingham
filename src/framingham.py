import numpy as np
import pandas as pd

def compute_framingham(df):
    """
    D'Agostino 2008 Framingham Risk Score
    Returns 10-year CVD risk as a probability (0-1)
    Only valid for ages 30-74
    """
    d = df.copy()

    # Rename for clarity
    age    = d['RIDAGEYR']
    sex    = d['RIAGENDR']       # 1=Male, 2=Female
    tchol  = d['LBXTC']          # total cholesterol mg/dL
    hdl    = d['LBDHDD']         # HDL mg/dL
    sbp    = d['BPXSY1']         # systolic BP
    smoke  = (d['SMQ020'] == 1).astype(float)   # 1=ever smoker
    diab   = (d['DIQ010'] == 1).astype(float)   # 1=diabetic

    # Filter to valid age range
    valid = (age >= 30) & (age <= 74)
    d = d[valid].copy()
    age   = age[valid]
    sex   = sex[valid]
    tchol = tchol[valid]
    hdl   = hdl[valid]
    sbp   = sbp[valid]
    smoke = smoke[valid]
    diab  = diab[valid]

    scores = []

    for i in d.index:
        a  = age[i]
        s  = sex[i]
        tc = tchol[i]
        h  = hdl[i]
        bp = sbp[i]
        sm = smoke[i]
        db = diab[i]

        if any(pd.isna([a, s, tc, h, bp, sm, db])):
            scores.append(np.nan)
            continue

        if s == 1:  # Male
            l = (3.06117 * np.log(a)
               + 1.12370 * np.log(tc)
               - 0.93263 * np.log(h)
               + 1.93303 * np.log(bp)
               + 0.65451 * sm
               + 0.57367 * db
               - 23.9802)
            risk = 1 - 0.88936 ** np.exp(l)

        else:  # Female
            l = (2.32888 * np.log(a)
               + 1.20904 * np.log(tc)
               - 0.70833 * np.log(h)
               + 2.76157 * np.log(bp)
               + 0.52873 * sm
               + 0.69154 * db
               - 26.1931)
            risk = 1 - 0.94833 ** np.exp(l)

        scores.append(float(risk))

    d['framingham_risk'] = scores
    d['high_risk'] = (d['framingham_risk'] >= 0.10).astype(int)
    return d


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.load_data import load_all
    clinical, paxhd, paxday, paxhr = load_all()
    result = compute_framingham(clinical)
    print(result[['SEQN','RIDAGEYR','RIAGENDR','framingham_risk','high_risk']].dropna().head(10))
    print(f"\nHigh risk (>=10%): {result['high_risk'].sum()} / {result['high_risk'].notna().sum()}")
    print(f"Mean risk: {result['framingham_risk'].mean():.3f}")