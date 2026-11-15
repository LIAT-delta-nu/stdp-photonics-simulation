import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from collections.abc import Callable
import matplotlib.colors as mc
import colorsys

# ============================================================
# Helper functions
# ============================================================
# Source - https://stackoverflow.com/a/49601444
# Posted by Ian Hincks, modified by community. See post 'Timeline' for change history
# Retrieved 2026-11-11, License - CC BY-SA 4.0

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    # import matplotlib.colors as mc
    # import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


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
    color=None,
    linestyle='-',
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
        color=color,
        linestyle=linestyle
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
    if label:
        ax.legend()
    if show:
        fig.show()
    if outfile:
        fig.savefig(outfile)
    else:
        pass

    return fig, ax

# ============================================================
# Plot probe gain
# ============================================================

def plot_probe_gain_dB(
    t,
    probe_gain_dB,
    signal_power_input=None,
    signal_power_output=None,
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
        zorder=3,
    )
    for signal_power, signal_label, color in zip([signal_power_input, signal_power_output], ["Signal input [a.u.]", "Signal output [a.u.]"], ["C1", "C2"]):
        if signal_power is not None:
            line_color= color
            fill_color=lighten_color(color)
            y_min, y_max = ax.get_ylim()
            P_min, P_max = min(signal_power), max(signal_power)
            signal_normalized = y_min + (signal_power - P_min) / (P_max - P_min) * (y_max - y_min) * 0.9
            ax.fill_between(
                t * 1e12, 
                y_min, 
                signal_normalized, 
                color=fill_color, 
                alpha=0.5,
                zorder=1,
            )
            ax.plot(
                t * 1e12, 
                signal_normalized, 
                color=line_color, 
                alpha=0.7, 
                label=signal_label,
                linewidth=0.8,
                # linestyle=':',
                zorder=2,
            )
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
    title="gm(n,λ)-α",
    show = False,
    outfile: str | Path = None, 
    figsize: tuple = (4.0, 4.0),
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_tight_layout(True)

    vmin = np.min(M) if np.min(M) < 0 else -1000 # irrelevant number since there are no M values below 0
    vmax = np.max(M) if np.max(M) > 0 else +1000

    custom_norm=TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

    im = ax.pcolormesh(
        N, Lambdas, M,
        cmap="RdBu_r",
        norm=custom_norm,
        # vmin=-np.nanmax(-M),
        # vmax=np.nanmax(M),
        shading="auto",
    )

    ax.set_xscale("log")
    ax.set_title(title)

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
    title="gm(n,λ)-α",
    figsize=(6.0,4.0),
    show=False,
    outfile=None,
):
    n_max_log10 = np.log10(n_max)
    n_min_log10 = np.log10(n_min) if n_min > 0 else 0

    ns_for_mesh = np.logspace(n_min_log10, n_max_log10, n_n)
    wavelengths_for_mesh = np.linspace(wavelength_min, wavelength_max, n_wavelength)

    Ns, WLs = np.meshgrid(ns_for_mesh, wavelengths_for_mesh)
    M = modal_gain(Ns, WLs) - alpha

    return plot_modal_gain_minus_alpha_mesh(Ns, WLs, M, title=title, show=show, outfile=outfile, figsize=figsize)
