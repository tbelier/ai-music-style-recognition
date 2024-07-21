#include "tools.h"


void print_data(std::string label, int value)
{
    std::cout << label << ": " << value << std::endl;
}

int read_n_bytes(std::ifstream &file, int n, bool is_signed)
{
    int word = 0;

    int byte;
    for (int k=0; k<n; k++)  // read a n*8 bits word
    {
        byte = file.get();
        word = word << 8 | static_cast<unsigned char>(byte);
    }

    // Convert to signed 16-bit if necessary
    if (is_signed && word >= 32768)  // 2^16 / 2
        word -= 65536;  // 2^16

    return word;
}

void twiddle_factors(std::array<Complex, FFT_SIZE / 2> &t)
{
    for (std::size_t k = 0; k < FFT_SIZE / 2; k++)
        t[k] = std::polar(1.0, -2.0 * PI * k / FFT_SIZE);
}

void bit_reverse_array(std::array<std::size_t, FFT_SIZE> &unscrambled)
{
    std::size_t m = std::log2(FFT_SIZE);
    for (std::size_t i = 0; i < FFT_SIZE; i++)
    {
        std::size_t j = i;

        j = (((j & 0xaaaaaaaa) >> 1) | ((j & 0x55555555) << 1));
        j = (((j & 0xcccccccc) >> 2) | ((j & 0x33333333) << 2));
        j = (((j & 0xf0f0f0f0) >> 4) | ((j & 0x0f0f0f0f) << 4));
        j = (((j & 0xff00ff00) >> 8) | ((j & 0x00ff00ff) << 8));
        j = ((j >> 16) | (j << 16)) >> (32 - m);

        if (i < j)
            unscrambled[i] = j;
        else
            unscrambled[i] = i;
    }
}

void ite_dit_fft(std::vector<Complex> &x)
{
    std::size_t problemSize = x.size();
    std::size_t stages = std::log2(problemSize);
    std::array<Complex, FFT_SIZE / 2> tf;
    twiddle_factors(tf);

    std::array<std::size_t, FFT_SIZE> unscrambled;
    bit_reverse_array(unscrambled);

    for (std::size_t i = 0; i < x.size(); i++)
    {
        std::size_t j = unscrambled[i];
        if (i < j)
            swap(x[i], x[j]);
    }

    for (std::size_t stage = 0; stage <= stages; stage++)
    {
        std::size_t currentSize = 1 << stage;
        std::size_t step = stages - stage;
        std::size_t halfSize = currentSize / 2;
        for (std::size_t k = 0; k < problemSize; k = k + currentSize)
        {
            for (std::size_t j = 0; j < halfSize; j++)
            {
                auto u = x[k + j];
                auto v = x[k + j + halfSize] * tf[j * (1 << step)];
                x[k + j] = (u + v);
                x[k + j + halfSize] = (u - v);
            }
        }
    }
}

void compute_mean_and_std(const std::vector<double>& values, double& mu, double& sigma)
{
    auto const count = static_cast<double>(values.size());

    double sum = 0.0;
    for (double v : values)
        sum += v;
    mu = sum / count;

    double sum_delta_square = 0.0;
    for (double v : values)
        sum_delta_square += std::pow(v - mu, 2.);
    sigma = std::sqrt(sum_delta_square / count);
}

void compute_descriptors(const std::string& file_name, std::vector<double>& mu_of_freq, std::vector<double>& std_of_freq)
{    
    std::ifstream file(file_name, std::ios::binary);

     if (!file.is_open())
        std::cerr << "Impossible d'ouvrir le fichier audio." << std::endl;

    // Lire l'en-tête du fichier audio
    int magic_number = read_n_bytes(file);
    int data_shift= read_n_bytes(file);
    int data_size = read_n_bytes(file);
    int encoding = read_n_bytes(file);
    int sample_rate = read_n_bytes(file);
    int n_channels = read_n_bytes(file);
    
    if (magic_number != AU_MAGIC)
        std::cerr << "Le fichier n'est pas au format AU." << std::endl;

    // print_data("Magical Number", magic_number);
    // print_data("Data Shift", data_shift);
    // print_data("Data Size", data_size);
    // print_data("Encoding", encoding);
    // print_data("Sample Rate", sample_rate);
    // print_data("Number of Channels", n_channels);

    file.seekg(data_shift, std::ios::beg);
    
    std::vector<std::vector<double>> vectorOfVectors(FFT_NUMBER, std::vector<double>(FFT_SIZE));  // 1288 arrays of size 512
    std::vector<Complex> vector512(FFT_SIZE);
    std::vector<double> vector512Normed(FFT_SIZE);
    
    for (int k=0; k < MAX_SAMPLE; k++)
    {   
        int i = k % FFT_SIZE; 
        int j = k / FFT_SIZE;

        vector512[i] = static_cast<Complex>(read_n_bytes(file, 2, true));
        
        if (i == FFT_SIZE-1)
        {
            // if we are the last value of the list we compute the fft
            ite_dit_fft(vector512);  // modify the list to add the fft instead of the temporal value
            
            // only keep norm squared
            for (int ind=0; ind < vector512.size(); ind++)
                vector512Normed[ind] = std::norm(vector512[ind]);  // std::norm computes the norm squared!

            // save it!
            vectorOfVectors[j] = vector512Normed; 
        }
    }

    file.close();
    
    double mu, sigma;
    std::vector<double> coefs_freq_i(FFT_NUMBER);
    for (int i=0; i < FFT_SIZE; i++)
    {
        for (int j=0; j < FFT_NUMBER; j++)
            coefs_freq_i[j] = vectorOfVectors[j][i];

        compute_mean_and_std(coefs_freq_i, mu, sigma);

        mu_of_freq[i]  = mu;
        std_of_freq[i] = sigma;
    }
}
