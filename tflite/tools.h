#ifndef TOOLS_H
#define TOOLS_H

#include "constants.h"
#include "mean_std.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <filesystem>
#include <string>

#include <array>
#include <vector>
#include <algorithm>

#include <complex>
#include <cmath>
#include <limits>

typedef std::complex<float> Complex;
extern std::vector<std::string> styles;

void print_data(std::string label, int value);
int read_n_bytes(std::ifstream &file, int n=4, bool is_signed=false);

void twiddle_factors(std::array<Complex, FFT_SIZE / 2> &t);
void bit_reverse_array(std::array<std::size_t, FFT_SIZE> &unscrambled);
void ite_dit_fft(std::vector<Complex> &x);

void compute_mean_and_std(const std::vector<double>& values, double& mu, double& sigma);

void compute_descriptors(const std::string& audio_file_name, std::vector<double>& mu_of_freq, std::vector<double>& std_of_freq);

#endif