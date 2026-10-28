"""3D visualization tools for polymer networks."""

import matplotlib.pyplot as plt


def plot_network(network):
    """Create a 3D plot of a polymer network."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = []
    y = []
    z = []

    for particle in network.particles:
        x.append(particle.position[0])
        y.append(particle.position[1])
        z.append(particle.position[2])  

    ax.scatter(x, y, z)

    for particle_a, particle_b in network.bonds:
        pos_a = network.particles[particle_a].position
        pos_b = network.particles[particle_b].position
        ax.plot(
            [pos_a[0], pos_b[0]],
            [pos_a[1], pos_b[1]],
            [pos_a[2], pos_b[2]],
            color="gray",
        )

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    return fig