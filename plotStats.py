import matplotlib.pyplot as plt
import numpy as np
from numflow import gershgorin_circles

from matplotlib.colors import hsv_to_rgb

def plot_stat(CC, RR, sub_index, ll_i):
    for i in range(len(CC)):
        c = np.array(CC[i])
        r = np.array(RR[i])
        ax[sub_index].fill_between(ll_i, np.absolute(c) - r, np.absolute(c) + r, alpha=0.2)
        ax[sub_index].plot(ll_i, np.absolute(c), "--")


def hinton(matrix):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    max_mag = np.abs(matrix).max()  # Max magnitude for square size scaling
    colormap = plt.cm.hsv  # Thermal colormap for phase

    for (x, y), value in np.ndenumerate(matrix):
        magnitude = np.abs(value)
        phase = np.angle(value)

        # Normalize magnitude and phase
        size = magnitude / max_mag
        color = colormap((phase + np.pi) / (2 * np.pi))  # Mapping phase to [0, 1]

        # Plot square
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    plt.xlim(-0.5, matrix.shape[0] - 0.5)
    plt.ylim(-0.5, matrix.shape[1] - 0.5)
    plt.gca().invert_yaxis()
    plt.show()




def plot_gershgorin_circles(matrix):
    n = matrix.shape[0]
    fig, ax = plt.subplots()
    centers, radii = gershgorin_circles(matrix)
    print(len(centers))
    for i in range(n):
        circle = plt.Circle((centers[i].real, centers[i].imag), radii[i], fill=False, color='blue',
                            label='GC' if i == 0 else "")
        ax.add_artist(circle)
        plt.plot(centers[i].real, centers[i].imag, '.', color='red')  # Center
    ax.set_xlim(-np.max(radii) - np.max(np.absolute(np.real(centers))), 0.2)
    ax.set_ylim(-np.max(radii) - np.max(np.absolute(np.imag(centers))),
                np.max(radii) + np.max(np.absolute(np.imag(centers))))
    ax.set_aspect('equal', 'box')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    # plt.title('Gershgorin Circles')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_brauer_circles(matrix):
    n = matrix.shape[0]
    fig, ax = plt.subplots()

    for i in range(n):
        for j in range(n):
            if i != j:
                center = matrix[i, i]
                radius = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i]) + np.abs(matrix[j, j] - matrix[i, i])
                circle = plt.Circle((center.real, center.imag), radius, fill=False, linestyle='--', color='green',
                                    label='Brauer Circle' if i == j - 1 else "")
                ax.add_artist(circle)
        plt.plot(matrix[i, i].real, matrix[i, i].imag, 'o', color='red')  # Center

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', 'box')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Brauer Circles')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_complex_matrix(M, figsize=(6, 6)):
    # Compute magnitude and angle
    mag = np.abs(M)
    angle = np.angle(M)

    # Normalize magnitude
    max_mag = mag.max() if mag.max() != 0 else 1
    saturation = mag / max_mag
    value = 1 - 0.7 * (mag / max_mag)  # zero -> white (v=1,s=0), large -> darker
    # Ensure value stays in [0,1]
    value = np.clip(value, 0, 1)

    # Hue from angle
    # Real axis (0 or ±π) → blue (hue=2/3)
    # Imag axis (±π/2) → red (hue=0)
    hue = (2 / 3) * (np.cos(angle) ** 2)

    # When mag=0, we want white. That means saturation=0 regardless of hue.
    saturation[mag == 0] = 0

    # Stack HSV
    hsv = np.dstack((hue, saturation, value))
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()