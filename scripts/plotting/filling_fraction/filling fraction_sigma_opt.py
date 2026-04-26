import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from collections import defaultdict

data_folder = '/home/duncan/data/repo/data/duncan'
plot_folder = '/home/duncan/data/repo/plots/filling_fraction/sigma'

def find_max_power_point(voltage_file):
    """Find voltage at maximum power point from IV curve."""
    df = pd.read_csv(voltage_file)
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors='coerce')
    
    V = df['sweep_parameter:V'].values
    J_CB = df['tot:current_CB:n_contact'].values
    J_IB = df['tot:current_IB:n_contact'].values
    J_VB = df['tot:current_VB:n_contact'].values
    J_tot = J_CB + J_IB + J_VB
    
    P = V * J_tot
    valid_mask = ~np.isnan(V) & ~np.isnan(P)
    max_idx = np.argmin(P[valid_mask])
    return V[valid_mask][max_idx]

def load_filling_fraction(data_file):
    """Load coord_x and f_IB from a data file."""
    if data_file.endswith('.csv.0'):
        data = pd.read_csv(data_file)
        coord_x = data['coord_x'].values
        u_IB = data['u_IB'].values
        N_IB = data['number_of_states_IB'].values
    elif data_file.endswith('.h5'):
        with h5py.File(data_file, 'r') as hf:
            geom = hf['/Mesh/0/mesh/geometry'][()]
            coord_x = geom[:, 0]
            u_IB = hf['/VisualisationVector/21'][()].flatten()
            N_IB = hf['/VisualisationVector/36'][()].flatten()
    else:
        return None, None

    valid = ~np.isnan(u_IB) & (N_IB > 0)
    return coord_x[valid], u_IB[valid] / N_IB[valid]

def find_mpp_data_file(sim_dir, mpp_voltage):
    """Find the data file closest to MPP voltage."""
    # Try HDF5 first
    h5_files = [f for f in sim_dir.glob('pd_V*.h5') if '_full' not in f.name]
    if h5_files:
        h5_voltages = [(float(f.stem.replace('pd_V=', '')), f) for f in h5_files]
        h5_voltages.sort(key=lambda x: abs(x[0] - mpp_voltage))
        return str(h5_voltages[0][1])
    
    # Fallback to CSV
    csv_files = list(sim_dir.glob('pd_V*.csv.0'))
    if csv_files:
        csv_voltages = []
        for f in csv_files:
            voltage_str = f.name.split('pd_V=')[1].split('.csv')[0]
            csv_voltages.append((float(voltage_str), f))
        csv_voltages.sort(key=lambda x: abs(x[0] - mpp_voltage))
        return str(csv_voltages[0][1])
    
    return None

def main():
    gcm_values = ['GCM_0', 'GCM_10', 'GCM_25', 'GCM_50']
    ci_iv_combos = [
        'ci_1.0e-13/iv_1.0e-13',
        'ci_1.0e-13/iv_5.0e-13',
        'ci_5.0e-13/iv_1.0e-13',
        'ci_5.0e-13/iv_5.0e-13',
    ]
    ci_iv_colors = {
        'ci_1.0e-13/iv_1.0e-13': 'blue',
        'ci_1.0e-13/iv_5.0e-13': 'orange',
        'ci_5.0e-13/iv_1.0e-13': 'green',
        'ci_5.0e-13/iv_5.0e-13': 'red',
    }
    
    # Group simulations: sims[Eg/mu_I][GCM][ci/iv] = (voltage_file, sim_dir)
    sims = defaultdict(lambda: defaultdict(dict))
    pd_v_files = list(Path(data_folder).rglob('pd_V.csv'))
    
    for voltage_file in pd_v_files:
        sim_dir = voltage_file.parent
        parts = sim_dir.relative_to(data_folder).parts
        if len(parts) < 6:
            continue
        plot_key = '/'.join(parts[:2])    # Eg/mu_I
        ci_iv_key = '/'.join(parts[2:4])  # ci/iv
        gcm = parts[4]                    # GCM_x
        
        sims[plot_key][gcm][ci_iv_key] = (voltage_file, sim_dir)
    
    for plot_key in sorted(sims.keys()):
        gcm_groups = sims[plot_key]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.suptitle(f'IB Filling Fraction — {plot_key.replace("/", ", ")}', fontsize=13)
        
        for idx, gcm in enumerate(gcm_values):
            ax = axes[idx // 2, idx % 2]
            ci_iv_data = gcm_groups.get(gcm, {})
            
            for ci_iv in ci_iv_combos:
                if ci_iv not in ci_iv_data:
                    continue
                
                voltage_file, sim_dir = ci_iv_data[ci_iv]
                mpp_voltage = find_max_power_point(str(voltage_file))
                data_file = find_mpp_data_file(sim_dir, mpp_voltage)
                
                if data_file is None:
                    continue
                
                coord_x, f_IB = load_filling_fraction(data_file)
                if coord_x is None:
                    continue
                
                ax.plot(coord_x, f_IB, marker='o', markersize=2, linewidth=1.2,
                        color=ci_iv_colors[ci_iv], alpha=0.7, label=ci_iv.replace('/', ', '))
            
            ax.set_title(f'{gcm}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        for ax in axes[1, :]:
            ax.set_xlabel(r'Position x ($\mu$m)', fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel(r'Filling Fraction $f_I$', fontsize=11)
        
        fig.tight_layout()
        
        output_dir = Path(plot_folder) / plot_key
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'filling_fraction_by_sigma.png'
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        
        print(f"Plotted: {plot_key}")


if __name__ == '__main__':
    main()