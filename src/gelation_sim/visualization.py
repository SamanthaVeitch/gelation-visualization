
"""3D visualization tools for polymer networks."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def plot_network(network, box_length=None):
    """
    Create a 3D visualization of a polymer network.

    Each particle is represented as a point at its position in
    3D space. Bonds between particles are represented as lines
    connecting their positions.

       Parameters
    ----------
    network : Network
        Polymer network containing particles and bonds.
        Each particle must have a position with x, y, and z
        coordinates. Bonds must contain pairs of particle indices.
    box_length : float, optional
        Length of the cubic simulation box in nanometers.
        When provided, all three axes use fixed limits from
        zero to the specified box length.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the 3D visualization of the network.
    """

    # Create a figure with a 3D plotting axis.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Assign a distinct color to each particle species.
    species_colors = {
        "monomer": "blue",
        "crosslinker": "orange",
        "initiator": "green",
    }

    # Collect the x, y, and z coordinates of each particle.
    x = []
    y = []
    z = []
    colors = []

    for particle in network.particles:
        x.append(particle.position[0])
        y.append(particle.position[1])
        z.append(particle.position[2])
        colors.append(species_colors[particle.species])

    # Plot particles as points in 3D space.
    ax.scatter(x, y, z, c=colors, depthshade=False)

    # Create a legend identifying each particle species.
    for species, color in species_colors.items():
        ax.scatter([], [], [], c=color, label=species)

    ax.legend(
        title="Species",
        loc="upper left",
        bbox_to_anchor=(0.8, 1.1)
    )

    # Draw a line between the positions of each bonded pair.
    for particle_a, particle_b in network.bonds:
        pos_a = network.particles[particle_a].position
        pos_b = network.particles[particle_b].position

        ax.plot(
            [pos_a[0], pos_b[0]],
            [pos_a[1], pos_b[1]],
            [pos_a[2], pos_b[2]],
            color="gray",
        )

    # Label coordinate axes using nanometers.
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    # Set fixed axis limits when a simulation box size is provided.
    if box_length is not None:
        ax.set_xlim(0, box_length)
        ax.set_ylim(0, box_length)
        ax.set_zlim(0, box_length)

    # Display the current simulation step.
    ax.set_title(f"Network at step {network.time}")

    # Return the figure for saving or further customization.
    return fig


def plot_snapshots(snapshots, output_dir):
    """
    Generate and save a 3D visualization for each network snapshot.

    Parameters
    ----------
    snapshots : list[Network]
        Network snapshots from different simulation steps.
    output_dir : str or Path
        Directory where the generated PNG files will be saved.

    Returns
    -------
    None
        Figures are saved to the output directory.
    """

    # Create the output directory if it does not exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save a figure for each simulation snapshot.
    for network in snapshots:
        fig = plot_network(network)

        filename = f"step_{network.time:06d}.png"
        fig.savefig(output_dir / filename, dpi=150)

        plt.close(fig)


def animate_snapshots(snapshots, output_file):
    """
    Create an animated GIF showing network evolution.

    Parameters
    ----------
    snapshots : list[Network]
        Network snapshots at different simulation steps.
    output_file : str or Path
        Filename for the generated GIF.

    Returns
    -------
    None
        The animation is saved to the specified file.
    """

    frames = []

    # Generate an image for each network snapshot.
    for network in snapshots:
        fig = plot_network(network)

        # Render the figure before capturing the image.
        fig.canvas.draw()

        # Convert the rendered figure into a Pillow image.
        frame = Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba()
        )

        frames.append(frame)
        plt.close(fig)

    # Save all frames as an animated GIF.
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )
