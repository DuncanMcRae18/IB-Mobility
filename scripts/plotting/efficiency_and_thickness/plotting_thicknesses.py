from matplotlib import pyplot as plt
import numpy as np
import sqlite3 as sql
import pandas as pd
import os
import yaml

database_path = "/home/duncan/data/main/database/planck.db"
plot_path = "/home/duncan/data/main/plots"

sqlite_connection = sql.connect(database_path)
df = pd.read_sql_query("SELECT * FROM TOTAL", sqlite_connection)

# Convert numeric columns from strings back to floats
numeric_cols = ['CB_E', 'IB_thickness', 'mu_I', 'sigma_opt_ci', 'sigma_opt_iv', 'IB_E']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

CB_E_vals = sorted(df['CB_E'].dropna().unique())
GCM_vals = sorted(df['GCM'].unique())
opt_ci_vals = sorted(df['sigma_opt_ci'].unique())
opt_iv_vals = sorted(df['sigma_opt_iv'].unique())

def main():
    for Eg in CB_E_vals:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        idx = 0
        
        for ci in opt_ci_vals:
            for iv in opt_iv_vals:
                ax = axes[idx]
                subset = df[(df['CB_E'] == Eg) & (df['sigma_opt_ci'] == ci) & (df['sigma_opt_iv'] == iv)]
                
                # Plot each GCM as a separate line
                for GCM in sorted(GCM_vals):
                    subset_gcm = subset[subset['GCM'] == GCM].sort_values('mu_I')
                    if not subset_gcm.empty:
                        ax.plot(np.log(subset_gcm['mu_I']), subset_gcm['IB_thickness'], marker='o', linestyle='-', label=f'GCM={GCM}')
                
                ax.set_title(f'sigma_ci={ci:.1e}, sigma_iv={iv:.1e}', fontsize=11)
                ax.set_xlabel('log(mu_I)', fontsize=10)
                ax.set_ylabel('IB Thickness', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                idx += 1
        
        fig.suptitle(f'Eg={Eg}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'IB_thickness_Eg{Eg}.png'), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()