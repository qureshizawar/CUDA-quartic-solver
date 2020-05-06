
#ifndef UTILS_H //using instead of #pragma once for greater compatibility
#define UTILS_H

#include <algorithm>
#include <random>
#include <iostream>
#include <ctime>

void tic();

void toc();

void generate_data(size_t size, int min, int max, float *data);

#endif