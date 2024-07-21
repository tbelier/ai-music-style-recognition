# EML-belbelgar

## Tasks

### DONE

- acquire audio signal from .au file
- extract descriptors (mean and standard deviation of FFT coefficients for 512 frequencies)
- SVM model
- decision tree & random forest
- ready for Raspberry Pi implementation
- neural network using Tensor Flow Lite
- implementation on a Raspberry Pi 4B

### TODO



## Help!

### Build the C++ project and extract the descriptors

- in the folder [WS] containing the CMakeLists.txt file : ```mkdir build/```
- enter the build/ folder: ```cd build/```
- build it: ```cmake .. & make```
- execute the extractor program: ```./au_feature_extractor```
- a file name "descriptors.csv" is created the folder [WS]

### Create, train and save the Machine Learning models

- run the following Python scripts : ```svm.py```, ```decision_tree.py```, ```random_forest.py```
- it generates the following header files : ```mean_std.h```, ```svm.h```, ```decision_tree.h```, ```random_forest.h```

### Make the inference from the audio folder

- make sure the C++ project has been compiled
- run one of the following programs : ```./svm_predict```, ```./decision_tree_predict```, ```./random_forest_predict```
