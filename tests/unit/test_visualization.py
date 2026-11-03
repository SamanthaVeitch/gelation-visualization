"""Unit tests for the gelation visualization module."""
import unittest
import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from tempfile import TemporaryDirectory  # noqa: E402

from PIL import Image  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from gelation_sim.model import Network, Particle  # noqa: E402
from gelation_sim.visualization import (
    plot_network,
    plot_snapshots,
    animate_snapshots,
)  # noqa: E402


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

    def test_plot_network_periodic_bonds(self):
        """Verify that periodic bonds are split at box boundaries."""

        particles = [
            Particle(0, "monomer", [0.2, 3.0, 3.0], 1),
            Particle(1, "monomer", [11.8, 3.0, 3.0], 1),
        ]

        network = Network(particles)
        network.add_bond(0, 1, step=0)

        fig = plot_network(network, box_length=12.0)
        ax = fig.axes[0]

        # The bond should be split into two short segments.
        self.assertEqual(len(ax.lines), 2)

        # Check the coordinates of the first segment.
        x, y, z = ax.lines[0].get_data_3d()

        self.assertAlmostEqual(x[0], 0.2)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertEqual(list(y), [3.0, 3.0])
        self.assertEqual(list(z), [3.0, 3.0])

        # Check the coordinates of the second segment.
        x, y, z = ax.lines[1].get_data_3d()

        self.assertAlmostEqual(x[0], 12.0)
        self.assertAlmostEqual(x[1], 11.8)
        self.assertEqual(list(y), [3.0, 3.0])
        self.assertEqual(list(z), [3.0, 3.0])

    def test_plot_network_multiple_periodic_boundaries(self):
        """Verify bonds crossing multiple boundaries are split correctly."""

        particles = [
            Particle(0, "monomer", [0.2, 0.4, 3.0], 1),
            Particle(1, "monomer", [11.8, 11.9, 3.0], 1),
        ]

        network = Network(particles)
        network.add_bond(0, 1, step=0)

        fig = plot_network(network, box_length=12.0)
        ax = fig.axes[0]

        # The bond crosses both X and Y boundaries.
        self.assertEqual(len(ax.lines), 3)

        # Check the first segment.
        x, y, z = ax.lines[0].get_data_3d()

        self.assertAlmostEqual(x[0], 0.2)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertAlmostEqual(y[0], 0.4)
        self.assertAlmostEqual(y[1], 0.15)

        # Check the second segment.
        x, y, z = ax.lines[1].get_data_3d()

        self.assertAlmostEqual(x[0], 12.0)
        self.assertAlmostEqual(x[1], 11.88)
        self.assertAlmostEqual(y[0], 0.15)
        self.assertAlmostEqual(y[1], 0.0)

        # Check the third segment.
        x, y, z = ax.lines[2].get_data_3d()

        self.assertAlmostEqual(x[0], 11.88)
        self.assertAlmostEqual(x[1], 11.8)
        self.assertAlmostEqual(y[0], 12.0)
        self.assertAlmostEqual(y[1], 11.9)

        # All segments should remain at Z = 3.0.
        for line in ax.lines:
            x, y, z = line.get_data_3d()
            self.assertEqual(list(z), [3.0, 3.0])

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

    def test_plot_snapshots_saves_multiple_figures(self):
        """Verify that each snapshot produces a PNG file."""

        particles = [
            Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
            Particle(1, "crosslinker", [1.0, 1.0, 1.0], 2),
        ]

        network = Network(particles, time=0)
        initial = network.snapshot()

        network.add_bond(0, 1, step=1)
        network.time = 1
        final = network.snapshot()

        with TemporaryDirectory() as directory:
            plot_snapshots([initial, final], directory)

            self.assertTrue(
                (Path(directory) / "step_000000.png").is_file()
            )
            self.assertTrue(
                (Path(directory) / "step_000001.png").is_file()
            )

    def test_animate_snapshots_creates_gif(self):
        """Verify that snapshots produce a multi-frame GIF."""

        particles = [
            Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
            Particle(1, "crosslinker", [1.0, 1.0, 1.0], 2),
        ]

        network = Network(particles)
        initial = network.snapshot()

        network.add_bond(0, 1, step=1)
        network.time = 1
        final = network.snapshot()

        with TemporaryDirectory() as directory:
            output = Path(directory) / "animation.gif"

            animate_snapshots([initial, final], output)

            self.assertTrue(output.is_file())

            with Image.open(output) as gif:
                self.assertEqual(gif.n_frames, 2)

    def test_plot_network_fixed_axis_limits(self):
        """Verify that the plot uses the specified simulation box size."""

        particles = [
            Particle(0, "monomer", [2.0, 3.0, 4.0], 1),
        ]

        network = Network(particles)

        fig = plot_network(network, box_length=12.0)
        ax = fig.axes[0]

        self.assertEqual(ax.get_xlim(), (0.0, 12.0))
        self.assertEqual(ax.get_ylim(), (0.0, 12.0))
        self.assertEqual(ax.get_zlim(), (0.0, 12.0))

    def tearDown(self):
        """Close all Matplotlib figures after each test."""
        plt.close("all")


if __name__ == '__main__':
    unittest.main()
