import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes import Axes


# ============================================================
# Plot carrier density
# ============================================================
def plot_carrier_density(
    t,
    z,
    n,
    outfile: str | Path = None, 
    ax: Axes = None,
    show: bool = None,
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
        plt.show()
    else:
        pass

    return fig, ax

# ============================================================
# Plot carrier density
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
    show: bool = None,
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
        plt.show()
    else:
        pass

    return fig, ax


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
        plt.show()
    else:
        pass

    if outfile:
        fig.savefig(outfile)

    return fig, ax

