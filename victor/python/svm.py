import numpy as np
from tools import load_dataset
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, label_names = load_dataset()

    # Create and fit the model
    svc = LinearSVC(C=1, loss='hinge')
    svc.fit(X_train, y_train)

    # Get coefficients
    svc_coefficient = svc.coef_  # (10, 1024)
    svc_intercept = svc.intercept_.reshape(-1, 1)  # (10,)

    # Compute decision function for the test input features
    decision_function_values = svc_coefficient @ X_test.T + svc_intercept

    # Predict the classes with the highest decision function values
    y_pred = np.argmax(decision_function_values, axis=0)
    y_pred = label_names[y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    confusion_mat = confusion_matrix(y_test, y_pred, labels=label_names)
    print("Confusion Matrix:")
    print(confusion_mat)

    classification_rep = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(classification_rep)

    # Open a C++ header file for writing
    header_filename = "../include/svm.h"
    with open(header_filename, "w") as header_file:
        # Write C++ code to declare, define, and initialize arrays
        header_file.write("#ifndef SVM_H\n")
        header_file.write("#define SVM_H\n\n")
        header_file.write("#include <vector>\n\n")

        # Declare and define the coefficient array
        header_file.write("const std::vector<std::vector<double>> svm_coefficient = {\n")
        for row in svc_coefficient:
            header_file.write("    {")
            header_file.write(", ".join(map(str, row)))
            header_file.write("},\n")
        header_file.write("};\n\n")

        # Declare and define the intersect array
        header_file.write("const std::vector<double> svm_intercept = {")
        header_file.write(", ".join(map(str, svc_intercept.flatten())))
        header_file.write("};\n\n")

        header_file.write("#endif // SVM_H\n")

    print(f"C++ header file '{header_filename}' generated successfully.")
