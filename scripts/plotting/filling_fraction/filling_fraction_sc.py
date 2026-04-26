import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from collections import defaultdict

data_folder = '/home/duncan/data/main/data/duncan'
plot_folder = '/home/duncan/data/main/plots/filling_fraction_sc'


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


def find_sc_data_file(sim_dir):
    """Find the data file closest to V=0 (short circuit)."""
    # Try CSV first
    csv_files = list(sim_dir.glob('pd_V=*.csv.0'))
    if csv_files:
        csv_voltages = []
        for f in csv_files:
            voltage_str = f.name.split('pd_V=')[1].split('.csv')[0]
            csv_voltages.append((float(voltage_str), f))
        csv_voltages.sort(key=lambda x: abs(x[0]))
        return str(csv_voltages[0][1])

    # Fallback to H5
    h5_files = [f for f in sim_dir.glob('pd_V=*.h5') if '_full' not in f.name]
    if h5_files:
        h5_voltages = [(float(f.stem.replace('pd_V=', '')), f) for f in h5_files]
        h5_voltages.sort(key=lambda x: abs(x[0]))
        return str(h5_voltages[0][1])

    return None


def main():
    mu_I_values = ['mu_I_1e-05', 'mu_I_0.001', 'mu_I_1', 'mu_I_30', 'mu_I_100']
    mu_I_colors = {
        'mu_I_1e-05': 'blue',
        'mu_I_0.001': 'orange',
        'mu_I_1': 'green',
        'mu_I_30': 'red',
        'mu_I_100': 'purple',
    }
    gcm_values = ['GCM_0', 'GCM_10', 'GCM_25', 'GCM_50']
    ci_iv_combos = [
        'ci_1.0e-13/iv_1.0e-13',
        'ci_1.0e-13/iv_5.0e-13',
        'ci_5.0e-13/iv_1.0e-13',
        'ci_5.0e-13/iv_5.0e-13',
    ]

    # Group: sims[Eg/ci_iv][GCM][mu_I] = sim_dir
    sims = defaultdict(lambda: defaultdict(dict))
    pd_v_files = list(Path(data_folder).rglob('pd_V.csv'))

    for voltage_file in pd_v_files:
        sim_dir = voltage_file.parent
        parts = sim_dir.relative_to(data_folder).parts
        if len(parts) < 6:
            continue
        eg, mu_I, ci, iv, gcm = parts[:5]
        plot_key = f'{eg}/{ci}/{iv}'
        sims[plot_key][gcm][mu_I] = sim_dir

    for plot_key in sorted(sims.keys()):
        eg, ci, iv = plot_key.split('/')
        gcm_groups = sims[plot_key]

        for idx, gcm in enumerate(gcm_values):
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(
                f'IB Filling Fraction at SC — {eg}, {ci}, {iv}, {gcm}'.replace('_', r'\_'),
                fontsize=13,
            )

            mu_I_data = gcm_groups.get(gcm, {})

            for mu_I in mu_I_values:
                if mu_I not in mu_I_data:
                    continue

                sim_dir = mu_I_data[mu_I]
                data_file = find_sc_data_file(sim_dir)

                if data_file is None:
                    continue

                coord_x, f_IB = load_filling_fraction(data_file)
                if coord_x is None:
                    continue

                label = mu_I.replace('mu_I_', r'$\mu_I$ = ')
                ax.plot(coord_x, f_IB, marker='o', markersize=2, linewidth=1.2,
                        color=mu_I_colors[mu_I], alpha=0.7, label=label)

            ax.set_xlabel(r'Position x ($\mu$m)', fontsize=11)
            ax.set_ylabel(r'Filling Fraction $f_I$', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            fig.tight_layout()

            output_dir = Path(plot_folder) / eg
            output_dir.mkdir(parents=True, exist_ok=True)
            ci_iv_safe = f'{ci}_{iv}'
            output_path = output_dir / f'filling_fraction_sc_{ci_iv_safe}_{gcm}.png'
            fig.savefig(str(output_path), dpi=150)
            plt.close(fig)

        print(f"Plotted: {plot_key}")


if __name__ == '__main__':
    main()
