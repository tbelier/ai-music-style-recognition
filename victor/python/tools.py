import numpy as np
from matplotlib import pyplot as plt


def load_dataset():
    # Chargement des données
    ds = np.genfromtxt('../descriptors.csv', delimiter=',', dtype=str)

    X = ds[:, :-1].astype(np.float64)  # Convertir les features en float
    y = ds[:, -1]  # Labels

    label_names = np.sort(np.unique(y))
    n_class = label_names.size

    # (700, n_features) & (700,)
    X_train = np.concatenate([X[100 * k:k * 100 + 70] for k in range(n_class)])
    y_train = np.concatenate([y[100 * k:k * 100 + 70] for k in range(n_class)])

    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    X_train = (X_train - mu) / sigma
    print('Train size:', y_train.size)

    # (300, n_features) & (300,)
    X_test = np.concatenate([X[70 + 100 * k:100 + k * 100] for k in range(n_class)])
    y_test = np.concatenate([y[70 + 100 * k:100 + k * 100] for k in range(n_class)])

    X_test = (X_test - mu) / sigma
    print('Test size:', y_test.size)

    # Save means and standard deviations
    # Open a C++ header file for writing
    header_filename = "../include/mean_std.h"
    with open(header_filename, "w") as header_file:
        # Write C++ code to declare, define, and initialize a vector<double>
        header_file.write("#ifndef MEAN_STD_H\n")
        header_file.write("#define MEAN_STD_H\n\n")
        header_file.write("#include <vector>\n\n")
        header_file.write("const std::vector<double> ds_mean = {")
        header_file.write(", ".join(map(str, mu)))
        header_file.write("};\n\n")
        header_file.write("const std::vector<double> ds_std = {")
        header_file.write(", ".join(map(str, sigma)))
        header_file.write("};\n\n")
        header_file.write("#endif // MEAN_STD_H\n")

    print(f"C++ header file '{header_filename}' generated successfully.")

    return X_train, y_train, X_test, y_test, label_names


def display_mean_and_std(X, y):
    n_class = np.unique(y).size
    occ_class = y.size // n_class

    plt.subplot(211)
    for k in range(n_class):
        Xk = np.mean(np.sqrt(X[k * occ_class:(k + 1) * occ_class, :512]), axis=0)
        yk = y[k * occ_class, 0]
        plt.plot(Xk, label=yk)

    plt.title('Mean')
    plt.yscale('log')
    plt.legend()

    plt.subplot(212)
    for k in range(n_class):
        Xk = np.mean(np.sqrt(X[k * occ_class:(k + 1) * occ_class, 512:]), axis=0)
        yk = y[k * occ_class, 0]
        plt.plot(Xk, label=yk)

    plt.title('Standard Deviation')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # (1000, 1025)
    dataset = np.genfromtxt('../data/descriptors.csv', delimiter=',', dtype=str)

    features = dataset[:, :-1].astype(np.float64)  # (1000, 1024)
    labels = dataset[:, -1:]  # (1000, 1)

    display_mean_and_std(features, labels)
