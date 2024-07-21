#include "tools.h"
#include "decision_tree.h"

#include <chrono>

using namespace std::chrono;
namespace fs = std::filesystem;

int main()
{   
    // Get starting timepoint
    auto start = high_resolution_clock::now();

    std::string folder_path = "../audio";

    std::vector<double> mu_of_freq(FFT_SIZE);
    std::vector<double> std_of_freq(FFT_SIZE);

    float descriptor[N_FEATURE];

    int n_au_file = 0;
    int n_good_prediction = 0;

    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        if (entry.is_regular_file())
        {
            n_au_file++;
            std::string file_name = entry.path().string();
            std::cout << " Traitement du fichier: " << file_name << std::endl;

            compute_descriptors(file_name, mu_of_freq, std_of_freq);

            // descriptor is the concatenation of mu_of_freq and std_of_freq
            for (int k_feature=0; k_feature < FFT_SIZE; k_feature++)
                descriptor[k_feature] = mu_of_freq[k_feature];
            for (int k_feature=FFT_SIZE; k_feature < N_FEATURE; k_feature++)
                descriptor[k_feature] = std_of_freq[k_feature - FFT_SIZE];
            
            // Normalize the descriptor vector
            for (int k_feature=0; k_feature < N_FEATURE; k_feature++)
                descriptor[k_feature] = (descriptor[k_feature] - ds_mean[k_feature]) / ds_std[k_feature];
            
            // using generated "inline" code for the decision tree
            const int32_t k_predicted = decision_tree_predict(descriptor, N_FEATURE);

            std::string predicted_class = styles[k_predicted];
            std::cout << "Predicted class: " << predicted_class << std::endl;

            if (file_name.find(predicted_class) != std::string::npos)
                n_good_prediction++;
        }
    }

    std::cout << "Score: " << n_good_prediction << " / " << n_au_file << std::endl;

    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Get duration. Substart timepoints to 
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);
 
    std::cout << "Time taken for inference: "
         << duration.count() / 1e6 << " seconds" << std::endl;

    return 0;
}
