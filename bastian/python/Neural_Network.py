import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models
import tensorflow.lite as lt
from sklearn.utils.class_weight import compute_class_weight

if __name__ == '__main__':
    # Charger les données
    dataset = np.genfromtxt('descriptors.csv', delimiter=',', dtype=str)
    X = dataset[:, :-1].astype(np.float64)  # (1000, 1024)
    y = dataset[:, -1]  # (1000,)

    # Convertir les labels en entiers avec Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Création du modèle
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(np.unique(y).size, activation='softmax'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Compute class weights
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

    # Create a dictionary mapping class indices to their respective weights
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model with class weights
    model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, class_weight=class_weight_dict)

    # Sauvegarde du modèle au format SavedModel
    model.save('saved_model')

    # Convertir le modèle en format TensorFlow Lite (.tflite)
    converter = lt.TFLiteConverter.from_saved_model('saved_model')
    tflite_model = converter.convert()

    # Sauvegarder le modèle TensorFlow Lite
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    # Évaluation du modèle sur l'ensemble de test
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)

    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)