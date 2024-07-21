import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # (1000, 1025)
    dataset = np.genfromtxt('descriptors.csv', delimiter=',', dtype=str)

    X = dataset[:, :-1].astype(np.float64)  # (1000, 1024)
    y = dataset[:, -1:]  # (1000, 1)

    n_class = 10
    occ_class = 100

    plt.subplot(211)
    for k in range(n_class):
        Xk = np.mean(np.sqrt(X[k*occ_class:(k+1)*occ_class, :512]), axis=0)
        yk = y[k*occ_class, 0]
        plt.plot(Xk, label=yk)

    plt.title('Mean')
    plt.yscale('log')
    plt.legend()

    plt.subplot(212)
    for k in range(n_class):
        Xk = np.mean(np.sqrt(X[k*occ_class:(k+1)*occ_class, 512:]), axis=0)
        yk = y[k*occ_class, 0]
        plt.plot(Xk, label=yk)

    plt.title('Standard Deviation')
    plt.yscale('log')
    plt.legend()
    plt.show()
