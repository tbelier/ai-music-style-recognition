cmake_minimum_required(VERSION 3.10)

project(eml_music_recognition LANGUAGES CXX)

set(CMAKE_BUILD_TYPE debug) # compilation en mode debug
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17") # options

# Add the include directory
include_directories(include)

add_executable( # création de l’exécutable binaire du projet
	au_feature_extractor # contient le nom du binaire
	src/main.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

add_executable( # création de l’exécutable binaire du projet
	random_forest_predict # contient le nom du binaire
	src/random_forest.cpp src/tools.cpp # liste des fichiers compilés pour le linking
)

# Include the 'include' directory for headers
target_include_directories(au_feature_extractor PRIVATE include)
target_include_directories(random_forest_predict PRIVATE include)
