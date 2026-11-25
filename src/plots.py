import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from collections.abc import Callable
import matplotlib.colors as mc
import colorsys
from functools import wraps
from matplotlib.lines import Line2D

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

# TODO: add the decorator to all plot functions
def standard_x_vs_y_plot(
    default_title : str = None, 
    default_x_label : str = None, 
    default_y_label : str = None
):
    def decorator(func):
        @wraps(func)
        def wrapper(
            *args,
            title : str = default_title,
            show : bool=False,
            outfile: str | Path = None,
            grid : bool = True,
            grid_minor : bool = True,
            xlabel : str = default_x_label,
            ylabel : str = default_y_label,
            ax: Axes = None,
            figsize: tuple = (4.0, 4.0),
            **kwargs
        ):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            fig.set_tight_layout(True)

            func(*args, ax=ax, **kwargs)

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)

            ax.tick_params(which="major", length=6, width=1)
            ax.tick_params(which="minor", length=3, width=0.7)

            if grid_minor:
                ax.minorticks_on()
                ax.grid(True, "both", alpha=0.3)
            if grid:
                ax.grid(True, "major", alpha=1.0)
            if show:
                fig.show()
            if outfile is not None:
                fig.savefig(outfile)
                plt.close(fig)

            return fig, ax

        return wrapper
    return decorator


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
        plt.close(fig)

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
        plt.close(fig)
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
        plt.close(fig)

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
        plt.close(fig)

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

# ============================================================
# Plot pulse energy vs delta t
# ============================================================

@standard_x_vs_y_plot(default_title="Pulse energy vs. Δt", default_y_label="Energy [μJ]", default_x_label="Δt [ps]")
def plot_energy_vs_delta_t( delta_t, energy, ax: Axes):
    energy_uJ = np.array(energy)*1e6
    delta_t_ps = np.array(delta_t)*1e12
    ax.plot(delta_t_ps, energy_uJ, marker='s')

# ============================================================
# Plot pulse peak power vs delta t
# ============================================================

@standard_x_vs_y_plot(default_title="Pulse peak power vs. Δt", default_y_label="Power [mWJ]", default_x_label="Δt [ps]")
def plot_energy_vs_delta_t(delta_t, energy, ax: Axes):
    energy_uJ = np.array(energy)*1e6
    delta_t_ps = np.array(delta_t)*1e12
    ax.plot(delta_t_ps, energy_uJ, marker='s')

@standard_x_vs_y_plot(default_y_label='Power [W]', default_x_label='time [ps]')
def plot_input_output(
    input : np.ndarray, 
    output : np.ndarray,  
    t : np.ndarray, 
    ax : Axes = None,
):

    plot_signal_output(
        t,
        input,
        label="Input",
        # outfile=save_dir / "forward_input_output.svg",
        ax=ax,
        show=False,
    )
    plot_signal_output(
        t,
        output,
        label="Output",
        # outfile=save_dir / "forward_input_output.svg",
        ax=ax,
        show=False,
    )

@standard_x_vs_y_plot(default_y_label='Power [W]', default_x_label='time [ps]', default_title='2 pulses input and output powers')
def plot_2_pulses(
    input_power_pulse_1 : np.ndarray,
    output_power_pulse_1 : np.ndarray,
    input_power_pulse_2 : np.ndarray,
    output_power_pulse_2 : np.ndarray,
    t : np.ndarray,
    ax : Axes = None,
):
    # Forward input and output filtered at pulses wavelengths
    pulse_1_output_color = 'b'
    pulse_1_input_color = lighten_color(pulse_1_output_color)
    pulse_2_output_color = 'r'
    pulse_2_input_color = lighten_color(pulse_2_output_color)
    input_linestyle = '--'
    output_linestyle = '-'
    plot_signal_output(
        t,
        input_power_pulse_1,
        label="Input pulse 1",
        # outfile=save_dir / "forward_input_output.svg",
        linestyle=input_linestyle,
        color=pulse_1_input_color,
        ax=ax,
        show=False,
    )

    plot_signal_output(
        t,
        output_power_pulse_1,
        ax=ax,
        label="Output pulse 1",
        # outfile=save_dir / "forward_input_output.svg",
        color=pulse_1_output_color,
        linestyle=output_linestyle,
        show=False,
    )

    plot_signal_output(
        t,
        input_power_pulse_2,
        ax=ax,
        label="Input pulse 2",
        # outfile=save_dir / "forward_input_output.svg",
        linestyle=input_linestyle,
        color=pulse_2_input_color,
        show=False,
    )

    plot_signal_output(
        t,
        output_power_pulse_2,
        ax=ax,
        label="Output pulse 2",
        title="",
        color=pulse_2_output_color,
        linestyle=output_linestyle,
        show=False,
    )

@standard_x_vs_y_plot(default_y_label='Power [W]', default_x_label='time [ps]', default_title='2 pulses input and output powers')
def plot_pulse_2_delta_t_sweep(
    ts_minus_t1: list[np.ndarray],
    output_powers_pulse_2: list[np.ndarray], 
    delta_t_peaks: list[float],
    ax : Axes = None,
):
    # Forward input and output filtered at pulses wavelengths
    t_peak_1_linestyle = '--'
    t_peak_2_linestyle = ':'

    t_peak_1_linewidth = 1.5
    t_peak_2_linewidth = 1.5
    
    pulse_1_color = 'k'

    output_linestyle = '-'

    ax.axvline(0, color=pulse_1_color, linewidth=t_peak_1_linewidth, linestyle=t_peak_1_linestyle)

    input_output_handles = [
        Line2D([0], [0], color=pulse_1_color, linestyle=t_peak_1_linestyle, linewidth=t_peak_1_linewidth, label="t_peak pulse 1"),
        Line2D([0], [0], color='k', linestyle=t_peak_2_linestyle, linewidth=t_peak_2_linewidth, label="t_peak pulse 2"),
    ]

    legend_input_output = ax.legend(
        handles=input_output_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    # Preserve the first legend
    ax.add_artist(legend_input_output)

    delta_t_handles = []

    for i, (
            t_minus_t1, 
            output_power_pulse_2, 
            delta_t_peak
        ) in enumerate(zip(
            ts_minus_t1,
            output_powers_pulse_2, 
            delta_t_peaks,
        )
    ):
        color_idx = i % 10
        pulse_2_output_color = f'C{color_idx}'
        
        delta_t_handles.append(
            Line2D([0], [0], color=pulse_2_output_color, lw=t_peak_2_linewidth, label=f"Δt={delta_t_peak*1e12:.0f} ps")
        )
        plot_signal_output(
            t_minus_t1,
            output_power_pulse_2,
            ax=ax,
            # label=f"Pulse 2 output (Δt_peak={delta_t_peak*1e12:.0f} ps)",
            # outfile=save_dir / "forward_input_output.svg",
            linestyle=output_linestyle,
            color=pulse_2_output_color,
            show=False,
        )
        ax.axvline(x=delta_t_peak*1e12, color=pulse_2_output_color, linewidth=t_peak_2_linewidth, linestyle=t_peak_2_linestyle)
    
    ax.legend(
        handles=delta_t_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )

    ax.figure.tight_layout()
