import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from simudo.example.fourlayer.sweep_extraction import SweepData

data_folder1 = '/home/duncan/data/repo/data/duncan/Eg_1.67/mu_I_0.001/ci_1.0e-13/iv_1.0e-13/GCM_0/0'
data_folder2 = '/home/duncan/data/repo/data/duncan/Eg_1.67/mu_I_30/ci_1.0e-13/iv_1.0e-13/GCM_0/0'
plot_folder = '/home/duncan/data/repo/plots/NR_recombination/norm_diff'

def plot(spatial001, spatial30):
    # spatial, IB_mask as returned by data.get_spatial_data and data.IB_mask
    
    CI_diff = (np.array(spatial30["g_nr_top_IB"])/np.mean(np.array(spatial30["g_nr_top_IB"])) - np.array(spatial001["g_nr_top_IB"])/np.mean(np.array(spatial001["g_nr_top_IB"])))
    IV_diff = (np.array(spatial30["g_nr_bottom_IB"])/np.mean(np.array(spatial30["g_nr_bottom_IB"])) - np.array(spatial001["g_nr_bottom_IB"])/np.mean(np.array(spatial001["g_nr_bottom_IB"])))

    plt.semilogy(
        np.array(spatial001["coord_x"]),
        np.abs(IV_diff),
        color="green",
        label=r"$|g_{ci}^{30}-g_{ci}^{0.001}|$",
    )
    plt.semilogy(
        np.array(spatial001["coord_x"]),
        np.abs(-IV_diff),
        color="red",
        label=r"$|g_{ci}^{0.001}-g_{iv}^{30}|$",
    )


def main():
    font = 12
    # plt.style.use('dark_background')
    sweep1 = SweepData(data_folder1)
    sweep2 = SweepData(data_folder2)
    spatial001 = sweep1.get_spatial_data(sweep1.mpp_row)
    spatial30 = sweep2.get_spatial_data(sweep2.mpp_row)
    IB_mask001 = sweep1.IB_mask(spatial001)
    IB_mask30 = sweep2.IB_mask(spatial30)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    plot(spatial001, spatial30)
    ax.set_xlabel(r'X position ($\mu$m)', fontsize=font)
    ax.set_ylabel(r'NR Recombination Difference $(\mathrm{cm}^{-3}\mathrm{s}^{-1})$', fontsize=font)
    ax.legend(fontsize=font)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(plot_folder) / 'nr_recomb_norm_diff.png'
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
