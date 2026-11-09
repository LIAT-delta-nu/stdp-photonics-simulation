import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from collections.abc import Callable

# ============================================================
# Plot carrier density
# ============================================================
def plot_carrier_density(
    t,
    z,
    n,
    outfile: str | Path = None, 
    ax: Axes = None,
    show: bool = False,
    figsize: tuple = (6, 4.0),
):
    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    fig.set_tight_layout(True)

    pcm = ax.pcolormesh(
        t*1e12, z*1e6, n,
        norm="log",
        shading="auto",
        # vmin=1e22,
        # vmax=1e26,
    )
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Position z (um)")
    fig.colorbar(pcm, ax=ax, label="Carrier density (m$^{-3}$)")
    ax.set_title("SOA carrier-density dynamics")

    if outfile:
        fig.savefig(outfile)

    if show:
        fig.show()
    else:
        pass

    return fig, ax

# ============================================================
# Plot signal output
# ============================================================

def plot_signal_output(
    t,
    signal_output,
    label=None,
    ylabel: str = 'Power [W]',
    xlabel: str = 'Time [ps]',
    grid : bool = True,
    grid_minor : bool = True,
    title : str = "SOA output",
    outfile: str | Path = None,
    ax: Axes = None,
    show: bool = False,
    figsize: tuple = (6, 4.0),
):
    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)

    fig.set_tight_layout(True)

    ax.plot(
        t * 1e12,
        signal_output,
        label=label,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.tick_params(which="major", length=6, width=1)
    ax.tick_params(which="minor", length=3, width=0.7)

    if grid_minor:
        ax.minorticks_on()
        ax.grid(True, "both", alpha=0.3)
    if grid:
        ax.grid(True, "major", alpha=1.0)
    if outfile:
        fig.savefig(outfile)
    if label:
        ax.legend()
    if show:
        fig.show()
    else:
        pass

    return fig, ax

# ============================================================
# Plot probe gain
# ============================================================

def plot_probe_gain_dB(
    t,
    probe_gain_dB,
    pump_output=None,
    outfile: str | Path=None,
    show=False,
    figsize: tuple = (6, 4.0),
):
    fig, ax = plt.subplots(figsize=figsize)

    fig.set_tight_layout(True)

    ax.plot(
        t * 1e12,
        probe_gain_dB,
        label="Probe gain",
    )

    if pump_output is not None:
        y_min, y_max = ax.get_ylim()
        P_min, P_max = min(pump_output), max(pump_output)
        pump_output_normalized = y_min + (pump_output - P_min) / (P_max - P_min) * (y_max - y_min) * 0.9
        ax.fill_between(t * 1e12, y_min, pump_output_normalized, color="lightgrey", alpha=0.5)
        ax.plot(t * 1e12, pump_output_normalized, color="grey", alpha=0.7, label="Pump output [a.u.]")
        ax.set_ylim(ymin=y_min, ymax=y_max)

    ax.legend()

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Probe gain (dB)")

    ax.set_title(f"SOA gain depletion and recovery")
    ax.grid()

    if show:
        fig.show()
    else:
        pass

    if outfile:
        fig.savefig(outfile)

    return fig, ax

# ============================================================
# Plot modal gain - losses
# ============================================================

def plot_modal_gain_minus_alpha_mesh(
    N, 
    Lambdas, 
    M,
    show = False,
    outfile: str | Path = None, 
    figsize: tuple = (4.0, 4.0),
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_tight_layout(True)

    custom_norm=TwoSlopeNorm(vmin=np.min(M), vcenter=0., vmax=np.max(M))

    im = ax.pcolormesh(
        N, Lambdas, M,
        cmap="RdBu_r",
        norm=custom_norm,
        # vmin=-np.nanmax(-M),
        # vmax=np.nanmax(M),
        shading="auto",
    )

    ax.set_xscale("log")
    ax.set_title("gm(n,λ)-α")

    fig.colorbar(im, ax=ax, label="m⁻¹")

    ax.set_xlabel("n")
    ax.set_ylabel("λ")

    cs = ax.contour(N, Lambdas, M, colors='k')
    ax.clabel(cs)

    ax.tick_params(axis="x", labelrotation=45)

    if outfile:
        fig.savefig(outfile)

    if show:
        fig.show()

    return fig, ax

def plot_modal_gain_minus_alpha(
    n_min, 
    n_max, 
    wavelength_min, 
    wavelength_max, 
    modal_gain: Callable[[float,float],float],
    alpha,
    n_n=50, 
    n_wavelength=50, 
    figsize=(6.0,4.0),
    show=False,
    outfile=None,
):
    n_min_log10 = np.log10(n_min)
    n_max_log10 = np.log10(n_max)

    ns_for_mesh = np.logspace(n_min_log10, n_max_log10, n_n)
    wavelengths_for_mesh = np.linspace(wavelength_min, wavelength_max, n_wavelength)

    Ns, WLs = np.meshgrid(ns_for_mesh, wavelengths_for_mesh)
    M = modal_gain(Ns, WLs) - alpha

    return plot_modal_gain_minus_alpha_mesh(Ns, WLs, M, show=show, outfile=outfile, figsize=figsize)
