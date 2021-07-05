
#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "MedianFilter.cuh"

bool MedianFilterParallelCPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int imRows, int imCols, int imChannels, uint32_t uKernelSize);
bool MedianFilterSequentialCPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int rows, int cols, int channels, uint32_t uKernelSize);

int main(int argc, char* argv[])
{
	cv::String sChaplinImagePath{ "H:/Projects/MedianFilter/images/Chaplin_noisy.png" };
	cv::String sLennaImagePath{ "H:/Projects/MedianFilter/images/Lenna_noisy.png" };
	uint32_t uKernelSize = 3;

	// -------------------------------------------------------------------------------- CPU

	// form input
	cv::Mat imageSrc = cv::imread(sChaplinImagePath, CV_8UC3);
	assert(imageSrc.isContinuous());
	std::vector<uint8_t> vImageSrc((uint8_t*)imageSrc.datastart, (uint8_t*)imageSrc.dataend);
	std::vector<uint8_t> vImageDstCPU(vImageSrc.size());

	// process image
	std::chrono::steady_clock::time_point timer_start, timer_end;
	timer_start = std::chrono::steady_clock::now();
	MedianFilterParallelCPU(vImageSrc, vImageDstCPU, imageSrc.rows, imageSrc.cols, imageSrc.channels(), uKernelSize);
	timer_end = std::chrono::steady_clock::now();
	std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count() << " milliseconds" << std::endl;

	// form output 
	cv::Mat imageDstCPU(imageSrc.size(), imageSrc.type());
	memcpy(imageDstCPU.data, vImageDstCPU.data(), vImageDstCPU.size());
	// -------------------------------------------------------------------------------- GPU

	// form input
	cv::Mat imageSrcGPU = cv::imread(sChaplinImagePath, CV_8UC3);
	assert(imageSrcGPU.isContinuous());
	std::vector<uint8_t> vImageSrcGPU((uint8_t*)imageSrcGPU.datastart, (uint8_t*)imageSrcGPU.dataend);
	std::vector<uint8_t> vImageDstGPU(vImageSrc.size());

	// process image
	timer_start = std::chrono::steady_clock::now();
	filter::MedianFilterGPU(vImageSrcGPU, vImageDstGPU, imageSrc.rows, imageSrc.cols, imageSrc.channels(), uKernelSize);
	timer_end = std::chrono::steady_clock::now();
	std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count() << " milliseconds" << std::endl;

	// form output
	cv::Mat imageDstGPU(imageSrcGPU.size(), imageSrcGPU.type());
	memcpy(imageDstGPU.data, vImageDstGPU.data(), vImageDstGPU.size());

	// -------------------------------------------------------------------------------- Show

	// show
	cv::imshow("src", imageSrc);
	cv::imshow("dst CPU", imageDstCPU);
	cv::imshow("dst GPU", imageDstGPU);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

//
// Parallel CPU worker function
//
bool MedianFilterCPUParallelThread(std::vector<uint8_t> const* imageSrc, std::vector<uint8_t>* imageDst, int minRow, int maxRow, int cols, int channels, uint32_t uKernelSize)
{
	int32_t uKernelRadius = uKernelSize / 2;
	int32_t uMedianIndex = (uKernelSize * uKernelSize - 1) / 2;
	std::vector<uint8_t> vKernelPixelValues(uKernelSize * uKernelSize);
	std::vector<uint8_t>::iterator vKernelPixelValuesIt;

	for (int32_t row = minRow; row < maxRow; ++row)
		for (int32_t col = uKernelRadius; col < cols - uKernelRadius; ++col)
			for (int32_t ch = 0; ch < channels; ++ch)
			{
				// get all pixel values
				int32_t i = -1;
				for (int32_t kernel_row = -uKernelRadius; kernel_row <= uKernelRadius; ++kernel_row)
					for (int32_t kernel_col = -uKernelRadius; kernel_col <= uKernelRadius; ++kernel_col)
						vKernelPixelValues[++i] = *(imageSrc->data() + channels * cols * (row + kernel_row) + channels * (col + kernel_col) + ch);

				// set median value
				std::sort(vKernelPixelValues.begin(), vKernelPixelValues.end());
				*(imageDst->data() + channels * cols * row + channels * col + ch) = vKernelPixelValues[uMedianIndex];
			}

	return true;
}

//
// Parallel CPU implementation
//
bool MedianFilterParallelCPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int rows, int cols, int channels, uint32_t uKernelSize)
{
	int32_t uKernelRadius = uKernelSize / 2;
	int32_t uMedianIndex = (uKernelSize * uKernelSize - 1) / 2;

	int n = std::thread::hardware_concurrency();
	if (n <= 0) throw n;

	// create threads
	std::vector<std::thread> threads(n);
	int rowStep = std::ceil(rows / n);
	threads[0] = std::thread(MedianFilterCPUParallelThread, &imageSrc, &imageDst, uKernelRadius, rowStep, cols, channels, uKernelSize);
	threads[n - 1] = std::thread(MedianFilterCPUParallelThread, &imageSrc, &imageDst, rowStep * (n - 1), rowStep * n - uKernelRadius, cols, channels, uKernelSize);
	for (int i = 1; i < n - 1; ++i)
		threads[i] = std::thread(MedianFilterCPUParallelThread, &imageSrc, &imageDst, (i * rowStep), (i + 1) * rowStep, cols, channels, uKernelSize);

	// wait
	for (auto& th : threads)
		th.join();

	return true;
}


//
// Sequential CPU implementation
//
bool MedianFilterSequentialCPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int rows, int cols, int channels, uint32_t uKernelSize)
{
	int32_t uKernelRadius = uKernelSize / 2;
	int32_t uMedianIndex = (uKernelSize * uKernelSize - 1) / 2;
	std::vector<uint8_t> vKernelPixelValues(uKernelSize * uKernelSize);

	// process (naive)
	for (int32_t row = uKernelRadius; row < rows - uKernelRadius; ++row)
		for (int32_t col = uKernelRadius; col < cols - uKernelRadius; ++col)
			for (int32_t ch = 0; ch < channels; ++ch)
			{
				// get all pixel values
				int32_t i = -1;
				for (int32_t kernel_row = -uKernelRadius; kernel_row <= uKernelRadius; ++kernel_row)
					for (int32_t kernel_col = -uKernelRadius; kernel_col <= uKernelRadius; ++kernel_col)
						vKernelPixelValues[++i] = *(imageSrc.data() + channels * cols * (row + kernel_row) + channels * (col + kernel_col) + ch);
				std::sort(vKernelPixelValues.begin(), vKernelPixelValues.end());

				// set median value
				*(imageDst.data() + channels * cols * row + channels * col + ch) = vKernelPixelValues[uMedianIndex];
			}

	return true;
}