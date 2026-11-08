import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Plot carrier density
# ============================================================
def plot_carrier_density(t, z, n, outfile: str | Path = None):
    fig, ax = plt.subplots()
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

    return fig, ax

def plot_signal_output(t, signal_output, outfile: str | Path = None):
    fig, ax = plt.subplots()

    ax.plot(
        t * 1e12,
        signal_output,
    )

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Output [W]")

    ax.set_title(f"SOA output")
    ax.grid()

    if outfile:
        fig.savefig(outfile)



def plot_probe_gain_dB(t, probe_gain_dB, outfile: str | Path = None):
    fig, ax = plt.subplots()

    ax.plot(
        t * 1e12,
        probe_gain_dB,
    )

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Probe gain (dB)")

    ax.set_title(f"SOA gain depletion and recovery")
    ax.grid()

    if outfile:
        fig.savefig(outfile)

