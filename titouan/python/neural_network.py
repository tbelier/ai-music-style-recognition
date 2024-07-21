import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = np.genfromtxt('descriptors.csv', delimiter=',', dtype=str)

X = dataset[:, :-1].astype(np.float64)  # Convertir les features en float
y = dataset[:, -1]  # Labels

labels = np.sort(np.unique(y))
n_class = labels.size

# Séparation des données en ensembles d'entraînement et de test
X_train = np.concatenate([X[100 * k:k * 100 + 70] for k in range(n_class)])
y_train = np.concatenate([y[100 * k:k * 100 + 70] for k in range(n_class)])

X_test = np.concatenate([X[70 + 100 * k:100 + k * 100] for k in range(n_class)])
y_test = np.concatenate([y[70 + 100 * k:100 + k * 100] for k in range(n_class)])

# Création d'un modèle séquentiel
model = Sequential()

# Ajout de la première couche cachée dense avec 32 neurones et fonction d'activation relu
model.add(Dense(32, activation='relu', input_shape=(100,)))  # 100 est la dimensionnalité de l'entrée

# Ajout d'une autre couche cachée
model.add(Dense(16, activation='relu'))

# Ajout de la couche de sortie avec 1 neurone (pour une tâche de régression)
model.add(Dense(1, activation='linear'))

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Résumé du modèle
model.summary()


# Données d'entrée et de sortie (exemple)
# Remplacez ces données par vos propres données
X_train = np.random.random((1000, 100))
y_train = np.random.random((1000, 1))

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32)
