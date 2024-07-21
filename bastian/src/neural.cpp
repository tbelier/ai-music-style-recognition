#include "tools.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <chrono>

using namespace std::chrono;
namespace fs = std::filesystem;

int main()
{
    // Get starting timepoint
    auto start = high_resolution_clock::now();

    std::string folder_path = "../audio";
    std::string model_path = "../model.tflite";

    // Load the TensorFlow Lite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Error loading the model." << std::endl;
        return -1;
    }

    // Create TensorFlow Lite interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder interpreter_builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    interpreter_builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Error creating the interpreter." << std::endl;
        return -1;
    }

    interpreter->AllocateTensors();  // Allocate memory for input and output tensors

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
            TfLiteTensor* input_tensor = interpreter->input_tensor(0);
            for (int i = 0; i < N_FEATURE; ++i)
            {
                input_tensor->data.f[i] = descriptor[i];
            }

            // Run the interpreter to perform the inference
            if (interpreter->Invoke() != kTfLiteOk) {
                std::cerr << "Error during inference." << std::endl;
                return -1;
            }

            // Get the result of the prediction
            TfLiteTensor* output_tensor = interpreter->output_tensor(0);
            const int32_t k_predicted = static_cast<int32_t>(output_tensor->data.f[0]);

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
