"""Unit tests for the gelation visualization module."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gelation_sim.model import Network, Particle
from gelation_sim.visualization import plot_network

def test_plot_network_creates_3d_figure():
    """Verify that plot_network returns a figure with a 3D axis."""

    network = Network([])

    fig = plot_network(network)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].name == "3d"

    plt.close(fig)


def test_plot_network_renders_particles():
    """Verify that all network particles are plotted."""

    particles = [
        Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
        Particle(1, "monomer", [1.0, 2.0, 3.0], 1),
        Particle(2, "crosslinker", [2.0, 3.0, 4.0], 2),
    ]

    network = Network(particles)

    fig = plot_network(network)
    ax = fig.axes[0]

    assert len(ax.collections) == 3

    plt.close(fig)