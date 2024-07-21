#include "tools.h"

namespace fs = std::filesystem;

int main()
{
    std::vector<std::string> styles = {"blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"};
    std::string folder_path = "../genres/";
    std::ofstream csv_file("../python/descriptors.csv");

    std::vector<double> mu_of_freq(FFT_SIZE);
    std::vector<double> std_of_freq(FFT_SIZE);

    int k_file = 0;
    for (const auto& style : styles)
    {
        std::string style_folder = folder_path + style;
        for (const auto& entry : fs::directory_iterator(style_folder))
        {
            if (entry.is_regular_file())
            {
                std::string file_name = entry.path().string();
                std::cout << '[' << ++k_file << " / 1000] Traitement du fichier: " << file_name << std::endl;

                compute_descriptors(file_name, mu_of_freq, std_of_freq);
                
                // Write means
                for (const auto& mu : mu_of_freq)
                    csv_file << mu << ",";

                // Write standard deviations
                for (const auto& std : std_of_freq)
                    csv_file << std << ",";

                // Write label and end line
                csv_file << style << std::endl;
            }
        }
    }

    csv_file.close();
    return 0;
}
