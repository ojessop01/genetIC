#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import pynbody

print("Python executable:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

# ============================================================
# Helper functions
# ============================================================

def limits_pm_3sigma(field):
    mean = np.mean(field)
    std  = np.std(field)
    return mean - 3.0 * std, mean + 3.0 * std

def plot_histogram(field, xlabel, filename, bins=200):
    data = field.ravel()
    mean = np.mean(data)
    std  = np.std(data)

    x = np.linspace(mean - 5 * std, mean + 5 * std, 1000)
    gaussian = (
        1.0 / (std * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std) ** 2)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, density=True, alpha=0.7, label="Simulation")
    ax.plot(x, gaussian, "k--", label="Gaussian ($\mu$, $\sigma$ matched)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.legend()
    plt.show()
    plt.savefig(filename)
    plt.close(fig)

def plot_midplane_triplet(f1, f2, diff, title1, title2, titlediff,
                          cbar_label, filename):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    vmin, vmax = limits_pm_3sigma(f1)
    im0 = axes[0].imshow(f1, origin="lower", cmap="RdBu_r",
                         vmin=vmin, vmax=vmax)
    axes[0].set_title(title1)
    fig.colorbar(im0, ax=axes[0], label=cbar_label)

    vmin, vmax = limits_pm_3sigma(f2)
    im1 = axes[1].imshow(f2, origin="lower", cmap="RdBu_r",
                         vmin=vmin, vmax=vmax)
    axes[1].set_title(title2)
    fig.colorbar(im1, ax=axes[1], label=cbar_label)

    vmin, vmax = limits_pm_3sigma(diff)
    im2 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
                         vmin=vmin, vmax=vmax)
    axes[2].set_title(titlediff)
    fig.colorbar(im2, ax=axes[2], label=r"$\Delta$" + cbar_label)
    plt.show()
    plt.savefig(filename)
    plt.close(fig)

# ============================================================
# Load ICs
# ============================================================

f1 = pynbody.load("/home/kbgj57/genetIC/example/iso/test.grafic_256/")
f2 = pynbody.load("/home/kbgj57/genetIC/example/no-iso/test.grafic_256/")

nx = int(f1._header["nx"])
ny = int(f1._header["ny"])
nz = int(f1._header["nz"])

deltab1 = f1["deltab"].reshape((nz, ny, nx))
deltac1 = f1["deltac"].reshape((nz, ny, nx))

deltab2 = f2["deltab"].reshape((nz, ny, nx))
deltac2 = f2["deltac"].reshape((nz, ny, nx))

# ============================================================
# Mid-plane slices
# ============================================================

z = nz // 2

db1 = deltab1[z]
db2 = deltab2[z]
dc1 = deltac1[z]
dc2 = deltac2[z]

db_diff = db1 - db2
dc_diff = dc1 - dc2

dbc1 = db1 - dc1
dbc2 = db2 - dc2
dbc_diff = dbc1 - dbc2

# ============================================================
# Mid-plane plots
# ============================================================

plot_midplane_triplet(
    db1, db2, db_diff,
    r"Test 1: $\delta_b$",
    r"Test 2: $\delta_b$",
    r"$\delta_b^{\mathrm{test\,1}} - \delta_b^{\mathrm{test\,2}}$",
    r"$\delta_b$",
    "deltab_midplane.pdf"
)

plot_midplane_triplet(
    dc1, dc2, dc_diff,
    r"Test 1: $\delta_c$",
    r"Test 2: $\delta_c$",
    r"$\delta_c^{\mathrm{test\,1}} - \delta_c^{\mathrm{test\,2}}$",
    r"$\delta_c$",
    "deltac_midplane.pdf"
)

plot_midplane_triplet(
    dbc1, dbc2, dbc_diff,
    r"Test 1: $\delta_{bc}$",
    r"Test 2: $\delta_{bc}$",
    r"$\delta_{bc}^{\mathrm{test\,1}} - \delta_{bc}^{\mathrm{test\,2}}$",
    r"$\delta_{bc}$",
    "deltabc_midplane.pdf"
)

# ============================================================
# Histograms: full 3D fields
# ============================================================

plot_histogram(deltab1, r"$\delta_b$ (Test 1)", "hist_deltab_test1.pdf")
plot_histogram(deltab2, r"$\delta_b$ (Test 2)", "hist_deltab_test2.pdf")
plot_histogram(deltab1 - deltab2,
               r"$\delta_b^{\mathrm{test\,1}}-\delta_b^{\mathrm{test\,2}}$",
               "hist_deltab_diff.pdf")

plot_histogram(deltac1, r"$\delta_c$ (Test 1)", "hist_deltac_test1.pdf")
plot_histogram(deltac2, r"$\delta_c$ (Test 2)", "hist_deltac_test2.pdf")
plot_histogram(deltac1 - deltac2,
               r"$\delta_c^{\mathrm{test\,1}}-\delta_c^{\mathrm{test\,2}}$",
               "hist_deltac_diff.pdf")

plot_histogram(deltab1 - deltac1,
               r"$\delta_{bc}$ (Test 1)",
               "hist_deltabc_test1.pdf")
plot_histogram(deltab2 - deltac2,
               r"$\delta_{bc}$ (Test 2)",
               "hist_deltabc_test2.pdf")
plot_histogram((deltab1 - deltac1) - (deltab2 - deltac2),
               r"$\delta_{bc}^{\mathrm{test\,1}}-\delta_{bc}^{\mathrm{test\,2}}$",
               "hist_deltabc_diff.pdf")
