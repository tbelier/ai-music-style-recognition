import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if __name__ == '__main__':
    # Chargement des données
    dataset = np.genfromtxt('descriptors.csv', delimiter=',', dtype=str)

    X = dataset[:, :-1].astype(np.float64)  # Convertir les features en float
    y = dataset[:, -1]  # Labels

    labels = np.sort(np.unique(y))
    n_class = labels.size

    # Création du modèle DecisionTree
    dt_clf = DecisionTreeClassifier(max_depth=1000)  # Vous pouvez ajuster les paramètres ici

    # Séparation des données en ensembles d'entraînement et de test
    X_train = np.concatenate([X[100 * k:k * 100 + 70] for k in range(n_class)])
    y_train = np.concatenate([y[100 * k:k * 100 + 70] for k in range(n_class)])

    X_test = np.concatenate([X[70 + 100 * k:100 + k * 100] for k in range(n_class)])
    y_test = np.concatenate([y[70 + 100 * k:100 + k * 100] for k in range(n_class)])

    print('Taille de l’entraînement:', y_train.size)
    print('Taille du test:', y_test.size)

    # Entraînement du modèle
    dt_clf.fit(X_train,y_train)

    # Prédiction
    y_pred = dt_clf.predict(X_test)


    import emlearn
    cmodel = emlearn.convert(dt_clf, method='inline')
    cmodel.save(file='titouan/include/decision_tree.h', name='decision_tree')
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision: {accuracy}")

    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    print("Matrice de confusion :")
    print(confusion_mat)

    classification_rep = classification_report(y_test, y_pred)
    print("Rapport de classification :")
    print(classification_rep)

