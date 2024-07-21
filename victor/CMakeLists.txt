cmake_minimum_required(VERSION 3.10)

project(eml_music_recognition LANGUAGES CXX)

set(CMAKE_BUILD_TYPE debug) # compilation en mode debug
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17") # options

# Add the include directory
include_directories(include)

add_executable( # création de l’exécutable binaire du projet
	au_feature_extractor # contient le nom du binaire
	src/extractor.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

add_executable( # création de l’exécutable binaire du projet
	svm_predict # contient le nom du binaire
	src/svm.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

add_executable( # création de l’exécutable binaire du projet
	decision_tree_predict # contient le nom du binaire
	src/decision_tree.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

add_executable( # création de l’exécutable binaire du projet
	random_forest_predict # contient le nom du binaire
	src/random_forest.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

# Link against the stdc++fs library (for old GCC compatibility)
# target_link_libraries(au_feature_extractor stdc++fs)
# target_link_libraries(svm_predict stdc++fs)
# target_link_libraries(decision_tree_predict stdc++fs)
# target_link_libraries(random_forest_predict stdc++fs)

# Include the 'include' directory for headers
target_include_directories(au_feature_extractor PRIVATE include)
target_include_directories(svm_predict PRIVATE include)
target_include_directories(decision_tree_predict PRIVATE include)
target_include_directories(random_forest_predict PRIVATE include)
