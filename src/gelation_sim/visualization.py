
"""3D visualization tools for polymer networks."""

import matplotlib.pyplot as plt


def plot_network(network):
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

    ax.legend(title="Species")

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

    # Return the figure for saving or further customization.
    return fig
