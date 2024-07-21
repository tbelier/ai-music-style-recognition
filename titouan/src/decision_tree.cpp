#include "tools.h"
#include "decision_tree.h"

namespace fs = std::filesystem;

int main()
{   
    std::ofstream csv_file("../python/descriptors.csv");
    std::string folder_path = "../audio/";
    std::vector<double> mu_of_freq(FFT_SIZE);
    std::vector<double> std_of_freq(FFT_SIZE);

    int k_file = 0;
    std::string style_folder = folder_path;
    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        if (entry.is_regular_file())
        {
            std::string file_name = entry.path().string();
            std::cout << " Traitement du fichier: " << file_name << std::endl;

            compute_descriptors(file_name, mu_of_freq, std_of_freq);
        }
    }

    float values[2*FFT_SIZE] = {};

    for (int i = 0; i < FFT_SIZE; ++i) {
        values[i] = mu_of_freq[i];
    }

    for (int i = 0; i < FFT_SIZE; ++i) {
        values[i + FFT_SIZE] = std_of_freq[i];
    }
    // using generated "inline" code for the decision tree
    const int32_t predicted_class = decision_tree_predict(values, FFT_SIZE*2);

    std::cout << predicted_class << std::endl;
    csv_file.close();
}
