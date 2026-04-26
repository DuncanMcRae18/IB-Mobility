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
numeric_cols = ['CB_E', 'eff', 'mu_I', 'sigma_opt_ci', 'sigma_opt_iv', 'IB_E']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

CB_E_vals = sorted(df['CB_E'].dropna().unique())
GCM_vals = sorted(df['GCM'].unique())
opt_ci_vals = sorted(df['sigma_opt_ci'].unique())
opt_iv_vals = sorted(df['sigma_opt_iv'].unique())

def main():
    font = 14
    plt.style.use('dark_background')
    for Eg in CB_E_vals:
        for ci in opt_ci_vals:
            for iv in opt_iv_vals:
                fig, ax = plt.subplots(figsize=(10, 6))
                subset = df[(df['CB_E'] == Eg) & (df['sigma_opt_ci'] == ci) & (df['sigma_opt_iv'] == iv)]
                
                # Plot each GCM as a separate line
                for GCM in sorted(GCM_vals):
                    subset_gcm = subset[subset['GCM'] == GCM].sort_values('mu_I')
                    if not subset_gcm.empty:
                        ax.semilogx(subset_gcm['mu_I'], subset_gcm['eff'], marker='o', linestyle='-', label=f'$m_G$={GCM}')
                
                ax.set_xlabel(r'$\mu_I \left(\frac{cm^2}{Vs}\right)$', fontsize=font)
                ax.set_xlabel(r'$\mu_I$', fontsize=font)
                ax.set_ylabel('Efficiency', fontsize=font)
                ax.tick_params(labelsize=font)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=font)
                
                plt.tight_layout()
                ci_str = f'{ci:.1e}'.replace('-', 'm')
                iv_str = f'{iv:.1e}'.replace('-', 'm')
                plt.savefig(os.path.join(plot_path, f'efficiency_Eg{Eg}_ci{ci_str}_iv{iv_str}.png'), dpi=150, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    main()