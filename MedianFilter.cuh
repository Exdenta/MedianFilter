#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

namespace filter {
	void MedianFilterGPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int rows, int cols, int channels, uint32_t iKernelSize);
}