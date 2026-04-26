import os
import sqlite3
import pandas as pd
import yaml

# sources
duncan_dir = '/home/duncan/data/main/data/duncan'
fatemeh_dir = '/home/duncan/data/fatemeh_mk'
fatemeh_results_dir = '/home/duncan/data/main/database/Fatemeh_results.csv'
archive_dir = '/home/duncan/data/main/archive'
planck_db = '/home/duncan/data/main/database/planck.db'

def parse_yaml(p):
    with open(p,'r') as f:
        return yaml.load(f, Loader=yaml.Loader)

def GCM_find(IB_E_val, CB_E_val):
    m167 = {0.58974918:0,1.08025082:0,0.55447173:10,1.04770392:10,0.49546686:25,1.00324958:25,0.37576187:50,0.93862021:50}
    m25 = {0.9739149949131315:0,0.9327205938861218:10,0.8622691192926665:25,0.710701104724266:50}
    try:
        IB = float(IB_E_val)
        CB = float(CB_E_val)
    except Exception:
        return ''
    if abs(CB-1.67)<1e-2:
        for ib,g in m167.items():
            if abs(ib-IB)<1e-3:
                return g
    if abs(CB-2.5)<1e-2:
        for ib,g in m25.items():
            if abs(ib-IB)<1e-3:
                return g
    return ''

def extract(folder,user):
    s=parse_yaml(os.path.join(folder,'submit.yaml'))
    o=parse_yaml(os.path.join(folder,'optimizer_summary.yaml'))
    r={}
    r.update(s.get('parameters',{}))
    r.update(o)
    r['folder_path']=folder
    log_path = os.path.join(folder,'info.log')
    if not os.path.exists(log_path):
        alt = os.path.join(os.path.dirname(folder), '0', 'info.log')
        if os.path.exists(alt):
            log_path = alt
    if os.path.exists(log_path):
        with open(log_path,'r') as f:
            first=f.readline()
            if 'Simudo version' in first:
                r['simudo_version']=first.split('Simudo version')[-1].strip()
    r['user']=user
    r['GCM']=GCM_find(r.get('IB_E'),r.get('CB_E'))
    return r

def extract_runs_to_df(base_dir,user):
    rows=[]
    for root,dirs,files in os.walk(base_dir):
        if 'submit.yaml' in files and 'optimizer_summary.yaml' in files:
            rows.append(extract(root,user))
    return pd.json_normalize(rows)

def extract_csv(p):
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'efficiency': 'eff',
        'sigma_iv': 'sigma_opt_iv',
        'sigma_ci': 'sigma_opt_ci'
    })
    df['user'] = 'Fatemeh'
    if 'CB_E' not in df.columns:
        df['CB_E'] = 1.67
    df['GCM'] = df.apply(lambda r: GCM_find(r.get('IB_E'), r.get('CB_E')), axis=1)
    # add empty simudo_version field for CSV rows
    df['simudo_version'] = ''
    df['folder_path'] = p
    return df

COLUMN_ORDER = [
    'CB_E', 'IB_E', 'GCM', 'mu_I',
    'sigma_opt_ci', 'sigma_opt_iv', 'eff', 'IB_thickness',
]

def write_df_to_db(db,table,df):
    import json
    _SQLITE_TYPES = (int, float, str, bytes, type(None))
    def _coerce(x):
        if isinstance(x, _SQLITE_TYPES): return x
        if isinstance(x, list): return json.dumps([str(v) for v in x])
        return str(x)
    df = df.copy()
    for col in df.columns:
        if not df[col].apply(lambda x: isinstance(x, _SQLITE_TYPES)).all():
            df[col] = df[col].apply(_coerce)
    # Reorder columns: priority columns first, then the rest
    priority = [c for c in COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]
    with sqlite3.connect(db) as conn:
        df.to_sql(table,conn,if_exists='replace',index=False)

if __name__=='__main__':
    print('Extracting good runs from Duncan data (Eg_1.67 and Eg_2.5)...')
    duncan_df = extract_runs_to_df(duncan_dir, 'Duncan')
    
    print(f'Found {len(duncan_df)} complete runs')
    
    if not duncan_df.empty:
        # Ensure metadata columns exist
        for extra in ('user', 'GCM', 'folder_path', 'simudo_version'):
            if extra not in duncan_df.columns:
                duncan_df[extra] = ''
        
        # Write to main table
        write_df_to_db(planck_db, 'runs', duncan_df)
        print(f'Wrote {len(duncan_df)} runs to {planck_db} (runs table)')
        
        # Also write to TOTAL table for compatibility
        write_df_to_db(planck_db, 'TOTAL', duncan_df)
        print(f'Wrote {len(duncan_df)} runs to {planck_db} (TOTAL table)')
    else:
        print('No complete runs found!')
    
    print(f'Database created at {planck_db}')
