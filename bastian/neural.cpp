#include "tools.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

#include <chrono>

using namespace std::chrono;
namespace fs = std::filesystem;

int main()
{
    // Get starting timepoint
    auto start = high_resolution_clock::now();

    std::string folder_path = "../audio";
    std::string model_path = "../model.h5";

    tensorflow::Session* session;
    tensorflow::SessionOptions session_options;
    tensorflow::NewSession(session_options, &session);

    // Load the neural network model
    tensorflow::GraphDef graph_def;
    tensorflow::Status status = ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
    if (!status.ok())
    {
        std::cerr << "Error loading the model: " << status.ToString() << std::endl;
        return -1;
    }

    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cerr << "Error creating the session: " << status.ToString() << std::endl;
        return -1;
    }

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
            std::cout << "Processing file: " << file_name << std::endl;

            compute_descriptors(file_name, mu_of_freq, std_of_freq);

            // descriptor is the concatenation of mu_of_freq and std_of_freq
            for (int k_feature = 0; k_feature < FFT_SIZE; k_feature++)
                descriptor[k_feature] = mu_of_freq[k_feature];
            for (int k_feature = FFT_SIZE; k_feature < N_FEATURE; k_feature++)
                descriptor[k_feature] = std_of_freq[k_feature - FFT_SIZE];

            // Normalize the descriptor vector
            for (int k_feature = 0; k_feature < N_FEATURE; k_feature++)
                descriptor[k_feature] = (descriptor[k_feature] - ds_mean[k_feature]) / ds_std[k_feature];

            // Create a tensor from the descriptor array
            tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, N_FEATURE}));
            auto input_tensor_mapped = input_tensor.tensor<float, 2>();
            for (int i = 0; i < N_FEATURE; ++i)
            {
                input_tensor_mapped(0, i) = descriptor[i];
            }

            // Define the name of the input and output nodes in the graph
            const std::string input_tensor_name = "dense";
            const std::string output_tensor_name = "dense_3";

            // Run the session to perform the inference
            std::vector<tensorflow::Tensor> output_tensors;
            status = session->Run({{input_tensor_name, input_tensor}}, {output_tensor_name}, {}, &output_tensors);
            if (!status.ok())
            {
                std::cerr << "Error during inference: " << status.ToString() << std::endl;
                return -1;
            }

            // Get the result of the prediction
            const auto& prediction = output_tensors[0].tensor<float, 2>();
            const int32_t k_predicted = static_cast<int32_t>(prediction(0, 0));

            std::string predicted_class = styles[k_predicted];
            std::cout << "Predicted class: " << predicted_class << std::endl;

            if (file_name.find(predicted_class) != std::string::npos)
                n_good_prediction++;
        }
    }

    std::cout << "Score: " << n_good_prediction << " / " << n_au_file << std::endl;

    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Get duration
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken for inference: " << duration.count() / 1e6 << " seconds" << std::endl;

    return 0;
}
