import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_xpt(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_sas(path, encoding='latin-1')

def load_all():
    print("Loading demographics...")
    demo  = load_xpt('DEMO_G.xpt')[['SEQN','RIDAGEYR','RIAGENDR','RIDRETH1']]

    print("Loading cholesterol...")
    tchol = load_xpt('TCHOL_G.xpt')[['SEQN','LBXTC']]
    hdl   = load_xpt('HDL_G.xpt')[['SEQN','LBDHDD']]

    print("Loading blood pressure...")
    bpx   = load_xpt('BPX_G.xpt')[['SEQN','BPXSY1','BPXDI1']]

    print("Loading labs...")
    glu   = load_xpt('GLU_G.xpt')[['SEQN','LBXGLU']]
    ghb   = load_xpt('GHB_G.xpt')[['SEQN','LBXGH']]

    print("Loading questionnaires...")
    smq   = load_xpt('SMQ_G.xpt')[['SEQN','SMQ020']]
    diq   = load_xpt('DIQ_G.xpt')[['SEQN','DIQ010']]

    print("Loading activity data...")
    paxhd  = load_xpt('PAXHD_G.xpt')
    paxday = load_xpt('PAXDAY_G.xpt')
    paxhr  = load_xpt('PAXHR_G.xpt')

    print("Merging clinical data...")
    clinical = (demo
        .merge(tchol, on='SEQN', how='left')
        .merge(hdl,   on='SEQN', how='left')
        .merge(bpx,   on='SEQN', how='left')
        .merge(glu,   on='SEQN', how='left')
        .merge(ghb,   on='SEQN', how='left')
        .merge(smq,   on='SEQN', how='left')
        .merge(diq,   on='SEQN', how='left')
    )

    print(f"Done. Clinical shape: {clinical.shape}")
    return clinical, paxhd, paxday, paxhr

if __name__ == '__main__':
    clinical, paxhd, paxday, paxhr = load_all()
    print(clinical.head())
