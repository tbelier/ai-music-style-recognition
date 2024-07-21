import emlearn
from tools import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, label_names = load_dataset()

    # Création du modèle RandomForest
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10)

    # Entraînement du modèle
    rf_clf.fit(X_train, y_train)

    # Prédiction
    y_pred = rf_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    confusion_mat = confusion_matrix(y_test, y_pred, labels=label_names)
    
    classification_rep = classification_report(y_test, y_pred)
    
    cmodel = emlearn.convert(rf_clf, method='inline')
    cmodel.save(file='../include/random_forest.h', name='random_forest')
    
    print(f"Précision: {accuracy}")
    print("Matrice de confusion :")
    print(confusion_mat)
    print("Rapport de classification :")
    print(classification_rep)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=label_names)
    disp.plot()
    plt.show()
