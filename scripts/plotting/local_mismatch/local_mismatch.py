import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from simudo.example.fourlayer.sweep_extraction import SweepData

data_folder = '/home/duncan/data/repo/data/duncan'
plot_folder = '/home/duncan/data/repo/plots/local_mismatch/individual'


def subgap_generation_mismatch_diagram(spatial, IB_mask, ax):
    mismatch = (
        spatial["g_opt_ci_IB"] + spatial["g_opt_iv_IB"]
    )  # the CI term is always negative
    
    # Plot directly to the provided axis
    ax.semilogy(
        spatial["coord_x"][IB_mask],
        mismatch[IB_mask],
        color="green",
        label=r"$g_{ci}-g_{iv}$",
    )
    ax.semilogy(
        spatial["coord_x"][IB_mask],
        -mismatch[IB_mask],
        color="red",
        label=r"$g_{iv}-g_{ci}$",
    )


def main():
    font = 12
    gcm_values = ['GCM_0', 'GCM_10', 'GCM_25', 'GCM_50']
    ci_iv_combos = [
        'ci_1.0e-13/iv_1.0e-13',
        'ci_1.0e-13/iv_5.0e-13',
        'ci_5.0e-13/iv_1.0e-13',
        'ci_5.0e-13/iv_5.0e-13',
    ]

    # Group simulations: sims[Eg/mu_I][ci/iv][GCM] = (voltage_file, sim_dir)
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

        sims[plot_key][ci_iv_key][gcm] = sim_dir

    for plot_key in sorted(sims.keys()):
        ci_iv_groups = sims[plot_key]

        for ci_iv in ci_iv_combos:
            gcm_data = ci_iv_groups.get(ci_iv, {})
            if not gcm_data:
                continue

            for gcm in gcm_values:
                # Skip if there's no data for this specific GCM
                if gcm not in gcm_data:
                    continue

                # Set up a new figure for each plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                fig.suptitle(
                    f'IB Optical Generation — {plot_key.replace("/", ", ")}, '
                    f'{ci_iv.replace("/", ", ")} | {gcm}',
                    fontsize=13,
                )

                sim_dir = gcm_data[gcm]
                sweep = SweepData(str(sim_dir))
                spatial = sweep.get_spatial_data(sweep.v_row(0))
                IB_mask = sweep.IB_mask(spatial)

                # Call the plotting function and pass the axis
                subgap_generation_mismatch_diagram(spatial, IB_mask, ax)

                # Formatting
                ax.set_xlabel(r'Position x ($\mu$m)', fontsize=font)
                ax.set_ylabel(r'Generation Mismatch $(\mathrm{cm}^{-3}\mathrm{s}^{-1})$', fontsize=font)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=font)

                fig.tight_layout()

                # Save the individual plot
                ci_iv_safe = ci_iv.replace('/', '_')
                output_dir = Path(plot_folder) / plot_key
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Include GCM in the filename to avoid overwriting
                output_path = output_dir / f'local_mismatch_{ci_iv_safe}_{gcm}.png'
                fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
                plt.close(fig)

                print(f"Plotted: {plot_key}, {ci_iv}, {gcm}")


if __name__ == '__main__':
    main()