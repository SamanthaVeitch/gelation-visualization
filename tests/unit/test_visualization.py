"""Unit tests for the gelation visualization module."""
import unittest
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from gelation_sim.model import Network, Particle  # noqa: E402
from gelation_sim.visualization import plot_network  # noqa: E402


class TestVisualization(unittest.TestCase):
    """Test the 3D visualization of polymer networks."""

    def test_plot_network_creates_3d_figure(self):
        """Verify that plot_network returns a figure with a 3D axis."""

        network = Network([])

        fig = plot_network(network)

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].name, "3d")

    def test_plot_network_renders_particles(self):
        """Verify that all network particles are plotted."""

        particles = [
            Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
            Particle(1, "monomer", [1.0, 2.0, 3.0], 1),
            Particle(2, "crosslinker", [2.0, 3.0, 4.0], 2),
        ]

        network = Network(particles)

        fig = plot_network(network)
        ax = fig.axes[0]

        self.assertEqual(len(ax.collections[0].get_offsets()), 3)

    def test_plot_network_renders_bonds(self):
        """Verify that bonds are drawn between connected particles."""

        particles = [
            Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
            Particle(1, "monomer", [1.0, 2.0, 3.0], 1),
        ]

        network = Network(particles)
        network.add_bond(0, 1, step=0)

        fig = plot_network(network)
        ax = fig.axes[0]

        self.assertEqual(len(ax.lines), 1)

        x, y, z = ax.lines[0].get_data_3d()

        self.assertEqual(list(x), [0.0, 1.0])
        self.assertEqual(list(y), [0.0, 2.0])
        self.assertEqual(list(z), [0.0, 3.0])

    def test_plot_network_colors_by_species(self):
        """Verify that particle colors correspond to their species."""

        particles = [
            Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
            Particle(1, "monomer", [1.0, 1.0, 1.0], 1),
            Particle(2, "crosslinker", [2.0, 2.0, 2.0], 2),
            Particle(3, "initiator", [3.0, 3.0, 3.0], 1),
        ]

        network = Network(particles)

        fig = plot_network(network)
        ax = fig.axes[0]

        # Render the figure so all particle colors are available.
        fig.canvas.draw()

        # Get the colors assigned to the particles.
        colors = ax.collections[0].get_facecolors()

        # Check that three species colors are present.
        self.assertEqual(len(set(map(tuple, colors))), 3)

        # Particles of the same species should have matching colors.
        self.assertTrue((colors[0] == colors[1]).all())

        # Different species should have different colors.
        self.assertFalse((colors[0] == colors[2]).all())
        self.assertFalse((colors[0] == colors[3]).all())
        self.assertFalse((colors[2] == colors[3]).all())

    def tearDown(self):
        """Close all Matplotlib figures after each test."""
        plt.close("all")


if __name__ == '__main__':
    unittest.main()
