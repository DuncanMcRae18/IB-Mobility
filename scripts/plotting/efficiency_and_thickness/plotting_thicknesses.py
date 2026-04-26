from matplotlib import pyplot as plt
import numpy as np
import sqlite3 as sql
import pandas as pd
import os
import itertools

# Paths
database_path = "/home/duncan/data/repo/data/duncan_results.db"
base_plot_path = "/home/duncan/data/repo/plots/thickness"
indiv_path = os.path.join(base_plot_path, "plots")
grid_path = os.path.join(base_plot_path, "subplots")

# Ensure directories exist
for p in [indiv_path, grid_path]:
    if not os.path.exists(p):
        os.makedirs(p)

sqlite_connection = sql.connect(database_path)
df = pd.read_sql_query("SELECT * FROM TOTAL", sqlite_connection)

# Data Cleaning
numeric_cols = ['CB_E', 'IB_thickness', 'mu_I', 'sigma_opt_ci', 'sigma_opt_iv']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

CB_E_vals = sorted(df['CB_E'].dropna().unique())
GCM_vals = sorted(df['GCM'].unique())
opt_ci_vals = sorted(df['sigma_opt_ci'].unique())
opt_iv_vals = sorted(df['sigma_opt_iv'].unique())

def main():
    font = 14
    sigma_combinations = list(itertools.product(opt_ci_vals, opt_iv_vals))

    for Eg in CB_E_vals:
        # --- GRID ---
        fig_grid, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig_grid.suptitle(f'Optimized IB Thickness Grid ($E_g$ = {Eg} eV)', fontsize=18)
        axes_flat = axes.flatten()

        for idx, (ci, iv) in enumerate(sigma_combinations):
            ax = axes_flat[idx]
            subset = df[(df['CB_E'] == Eg) & (df['sigma_opt_ci'] == ci) & (df['sigma_opt_iv'] == iv)]
            
            # --- INDIVIDUAL ---
            fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
            
            for GCM in GCM_vals:
                subset_gcm = subset[subset['GCM'] == GCM].sort_values('mu_I')
                if not subset_gcm.empty:
                    line_params = {'marker': 'o', 'linestyle': '-', 'label': f'$m_G$={GCM}'}
                    ax.semilogx(subset_gcm['mu_I'], subset_gcm['IB_thickness'], markersize=4, **line_params)
                    ax_ind.semilogx(subset_gcm['mu_I'], subset_gcm['IB_thickness'], **line_params)

            # Individual Plot Finalize
            ax_ind.set_title(f'$E_g$={Eg} eV | $\sigma_{{ci}}$={ci:.1e} | $\sigma_{{iv}}$={iv:.1e}', fontsize=font)
            ax_ind.set_xlabel(r'$\mu_I$ ($cm^2/Vs$)', fontsize=font)
            ax_ind.set_ylabel('IB Thickness ($\mu$m)', fontsize=font)
            ax_ind.grid(True, alpha=0.3); ax_ind.legend(fontsize=font-2)
            
            ci_s = f'{ci:.1e}'.replace('-', 'm'); iv_s = f'{iv:.1e}'.replace('-', 'm')
            fig_ind.savefig(os.path.join(indiv_path, f'thick_Eg{Eg}_ci{ci_s}_iv{iv_s}.png'), dpi=150)
            plt.close(fig_ind)

            # Subplot Grid Finalize
            ax.set_title(f'$\sigma_{{ci}}$={ci:.1e}, $\sigma_{{iv}}$={iv:.1e}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if idx == 0: ax.legend(fontsize=10)

        # Grid Plot Finalize
        fig_grid.text(0.5, 0.02, r'$\mu_I$ ($cm^2/Vs$)', ha='center', fontsize=16)
        fig_grid.text(0.02, 0.5, 'IB Thickness ($\mu$m)', va='center', rotation='vertical', fontsize=16)
        fig_grid.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        fig_grid.savefig(os.path.join(grid_path, f'thick_grid_Eg{Eg}.png'), dpi=200)
        plt.close(fig_grid)
        print(f"Completed Thickness suite for Eg={Eg}")

if __name__ == "__main__":
    main()