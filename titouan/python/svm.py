import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


if __name__ == '__main__':
    # (1000, 1025)
    dataset = np.genfromtxt('descriptors.csv', delimiter=',', dtype=str)

    X = dataset[:, :-1].astype(np.float64)  # (1000, 1024)
    y = dataset[:, -1]  # (1000,)

    labels = np.sort(np.unique(y))
    n_class = labels.size

    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss='hinge')),  # custom C HERE
    ])

    X_train = np.concatenate([X[100 * k:k * 100 + 70] for k in range(n_class)])
    y_train = np.concatenate([y[100 * k:k * 100 + 70] for k in range(n_class)])

    X_test = np.concatenate([X[70 + 100 * k:100 + k * 100] for k in range(n_class)])
    y_test = np.concatenate([y[70 + 100 * k:100 + k * 100] for k in range(n_class)])

    print('Train size:', y_train.size)
    print('Test size:', y_test.size)

    svm_clf.fit(X_train, y_train)

    # Prediction
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(confusion_mat)

    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)
