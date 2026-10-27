"""3D visualization tools for polymer networks."""

import matplotlib.pyplot as plt


def plot_network(network):
    """Create a 3D plot of a polymer network."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    return fig