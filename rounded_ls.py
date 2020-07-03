import numpy as np
from scipy import sparse


def RLS(psi: np.ndarray):
    assert psi.ndim == 2, "Input x should be a 2d array!"

    M, N = psi.shape
    Lx = sparse.diags([-1, 1], [0, 1], shape=(N - 1, N), format='csc')
    Ly = sparse.diags([-1, 1], [0, 1], shape=(M - 1, M), format='csc')

    delta_psi_x = psi @ Lx.T
    delta_psi_y = Ly @ psi
    delta_k_x = -np.round(delta_psi_x * 0.5 / np.pi)
    delta_k_y = -np.round(delta_psi_y * 0.5 / np.pi)

    A = Ly.T @ Ly
    B = Lx.T @ Lx
    C = Ly.T @ delta_k_y + delta_k_x @ Lx
    U, s, Vh = np.linalg.svd(B.toarray())
    D = C @ U
    I = sparse.eye(M)
    A = A.toarray()

    k = np.empty_like(psi)
    for i in range(N):
        k[:, i] = np.linalg.inv(A + I * s[i]) @ D[:, i]

    k_hat = np.round(k @ U.T)
    result = psi + 2 * np.pi * k_hat
    return result


if __name__ == '__main__':
    from utils import wrap_func
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mcf import mcf_nx, mcf_or
    from graph_cuts import puma, puma_or
    from skimage.restoration import unwrap_phase
    from skimage import data, img_as_float, color, exposure

    N = 200
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    x = xx / N
    sigma = yy / N * np.pi / 5
    plane = N * 0.5 * np.pi * x ** 2
    plane += np.random.randn(*plane.shape) * sigma
    wrapped_plane = wrap_func(plane)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sigma, x, plane, cmap=plt.cm.coolwarm)
    ax.set_xlabel('sigma')
    ax.set_ylabel('X')
    plt.show()

    result, elist = puma_or(wrapped_plane)
    #result = mcf_or(wrapped_plane)
    result -= result.min()

    #plt.plot(elist)
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sigma, x, result, cmap=plt.cm.coolwarm)
    ax.set_xlabel('sigma')
    ax.set_ylabel('X')
    plt.show()

    error = plane - result
    plt.imshow(error, aspect='auto')
    plt.colorbar()
    plt.show()