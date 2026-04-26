import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

data_folder = '/home/duncan/data/main/data/duncan'
plot_folder = '/home/duncan/data/main/plots/filling_fraction'

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
    mpp_voltage = V[valid_mask][max_idx]
    
    return mpp_voltage

def plot_filling_fraction(data_file, sim_folder, plot_base_folder):
    """Plot IB filling fraction at max power point."""
    
    print(data_file)

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
        print(f"Unsupported file format: {data_file}")
        return

    # Only keep IB region (where number_of_states > 0 and u_IB is valid)
    valid = ~np.isnan(u_IB) & (N_IB > 0)
    coord_x = coord_x[valid]
    f_IB = u_IB[valid] / N_IB[valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coord_x, f_IB, marker='o', markersize=3, linewidth=1.5, 
            color='blue', label='IB Filling Fraction $f_I$', alpha=0.7)

    ax.set_xlabel(r'Position x ($\mu$m)', fontsize=12)
    ax.set_ylabel(r'Filling Fraction $f_I$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()

    # Create nested folder structure
    sim_path = Path(sim_folder)
    data_base = Path(data_folder)
    relative_path = sim_path.relative_to(data_base)
    output_dir = Path(plot_base_folder) / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'filling_fraction_vs_x.png'
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    
def main():
    """Process all simulations: plot filling fraction at maximum power point."""
    pd_v_files = list(Path(data_folder).rglob('pd_V.csv'))
    
    for voltage_file in pd_v_files:
        mpp_voltage = find_max_power_point(str(voltage_file))
        sim_dir = voltage_file.parent
        
        # Find data file closest to MPP voltage
        # Try HDF5 first
        h5_files = [f for f in sim_dir.glob('pd_V*.h5') if '_full' not in f.name]
        if h5_files:
            h5_voltages = [(float(f.stem.replace('pd_V=', '')), f) for f in h5_files]
            h5_voltages.sort(key=lambda x: abs(x[0] - mpp_voltage))
            data_file = str(h5_voltages[0][1])
            plot_filling_fraction(data_file, str(sim_dir), plot_folder)
            continue
        
        # Fallback to CSV
        csv_files = list(sim_dir.glob('pd_V*.csv.0'))
        if csv_files:
            csv_voltages = []
            for f in csv_files:
                # Extract voltage from name like "pd_V=0.029333333333333.csv.0"
                voltage_str = f.name.split('pd_V=')[1].split('.')[0]
                csv_voltages.append((float(voltage_str), f))
            csv_voltages.sort(key=lambda x: abs(x[0] - mpp_voltage))
            data_file = str(csv_voltages[0][1])
            plot_filling_fraction(data_file, str(sim_dir), plot_folder)
            continue
        
        print(f"No data file found in {sim_dir.name}")


if __name__ == '__main__':
    main()