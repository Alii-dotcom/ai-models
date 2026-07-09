import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt
plt.ion()

def get_w_h(n: int, max_ratio: int = 2):
    values, rests = [], []
    r = int(np.sqrt(n))
    for i in reversed(range(1, r + 1)):
        if (n // i) / i > max_ratio and i < r:
            break
        rests.append(n % i)
        values.append(n // i)
    return r - np.argmin(rests), values[np.argmin(rests)]

def show_eigenfaces(eigenfaces: np.ndarray, size: Tuple,
                    max_components: int = 25):
    """
    Plots ghostly eigenfaces.
    
    Parameters
    ----------
    eigenfaces: ndarray
        eigenfaces (eigenvectors of face covariance matrix).
    size: tuple
        the size of each face image like (h, w).

    Returns
    -------
    None
    """

    num_eigenfaces = eigenfaces.shape[1]
    (w, h) = get_w_h(min(max_components, num_eigenfaces))

    fig, ax = plt.subplots(nrows=h, ncols=w, sharex='col',
                           sharey='row', figsize=(5, 5))

    # normalize `ax` to a 2-D ndarray of shape (h, w) so indexing is consistent
    ax = np.array(ax, ndmin=2)
    try:
        ax = ax.reshape(h, w)
    except Exception:
        ax = np.reshape(ax, (h, w))

    total_axes = h * w
    # iterate over all subplot positions, but only draw available eigenfaces
    for idx in range(total_axes):
        i, j = divmod(idx, w)
        if idx < num_eigenfaces:
            f = np.array(eigenfaces[:, idx])
            f = np.reshape(f, shape=size)
            ax[i, j].imshow(f, cmap='gray')
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            ax[i, j].set_title(f"eig {idx}")
            ax[i, j].set_aspect('equal')
        else:
            ax[i, j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.4)

def show_3d_faces_with_class(points: np.ndarray, labels: np.ndarray):
    """
    Plots 3d data in colorful point (color is class).
    
    Parameters
    ----------
    points: ndarray
        3d points to plot (shape: (n_samples, 3)).
    labels: ndarray
        classes (shape: (n_samples,)).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, c=labels, s=60)
    plt.show(block=True)


def show_nearest_neighbor(X_train: np.array, Y_train: np.ndarray,
                          X_test: np.array, Y_test: np.ndarray,
                          nearest_neighbors: np.array):

    # visualize nearest neighbors
    _, (ax0, ax1) = plt.subplots(1, 2)

    while True:
        # extract random index
        test_idx = np.random.randint(0, X_test.shape[0])

        X_cur, Y_cur = X_test[test_idx], Y_test[test_idx]
        X_cur_pred, Y_cur_pred = X_train[nearest_neighbors[test_idx]], Y_train[nearest_neighbors[test_idx]]

        ax0.imshow(np.reshape(X_cur, shape=(112, 92)), cmap='gray')
        ax0.set_title(f'Test face - ID {int(Y_cur)}')

        color = 'r' if Y_cur != Y_cur_pred else 'g'

        ax1.imshow(np.reshape(X_cur_pred, shape=(112, 92)), cmap='gray')
        ax1.set_title(f'Nearest neighbor - ID {int(Y_cur_pred)}', color=color)

        # plot faces
        plt.waitforbuttonpress()

