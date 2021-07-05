#include "MedianFilter.cuh"

__device__ void csort(uint8_t ls[], int l, int r);

//
// GPU implementation
//
__global__ void MedianFilterGPUKernel(uint8_t* imageSrc, uint8_t* imageDst, int channels, int32_t uKernelRadius, uint32_t uKernelPixelCount, uint32_t uMedianIdx)
{
	int row = blockIdx.x;
	int col = blockIdx.y;
	int rows = gridDim.x;
	int cols = gridDim.y;

	//if (row > 0 && col > 0 && row < rows && col < cols)
	//{

	auto pPixelValues = new uint8_t[uKernelPixelCount];

	// get all pixel values
	int32_t i = 0;
	for (int32_t kernel_row = -uKernelRadius; kernel_row <= uKernelRadius; ++kernel_row)
		for (int32_t kernel_col = -uKernelRadius; kernel_col <= uKernelRadius; ++kernel_col)
			pPixelValues[++i] = imageSrc[(row + kernel_row) * cols + (col + kernel_col)];

	csort(pPixelValues, 0, uKernelPixelCount);
	imageDst[row * cols + col] = pPixelValues[uMedianIdx];
	//}
}


namespace filter
{
	void MedianFilterGPU(std::vector<uint8_t> const& imageSrc, std::vector<uint8_t>& imageDst, int rows, int cols, int channels, uint32_t uKernelSize)
	{
		int32_t uKernelRadius = uKernelSize / 2;
		uint32_t uKernelPixelCount = uKernelSize * uKernelSize;
		uint32_t uMedianIdx = (uKernelPixelCount + 1) / 2;

		uint8_t* dev_imageSrc;
		uint8_t* dev_imageDst;

		// allocate the memory on the GPU
		cudaMalloc((void**)&dev_imageSrc, imageSrc.size() * sizeof(uint8_t));
		cudaMalloc((void**)&dev_imageDst, imageDst.size() * sizeof(uint8_t));

		// fill the arrays ‘imageSrc’ and ‘imageDst’ on the GPU
		cudaMemcpy(dev_imageSrc, imageSrc.data(), imageSrc.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_imageDst, imageDst.data(), imageDst.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

		// run filter
		dim3 grid(rows, cols);
		MedianFilterGPUKernel << < grid, 1 >> > (dev_imageSrc, dev_imageDst, channels, uKernelRadius, uKernelPixelCount, uMedianIdx);

		// copy data from GPU
		cudaMemcpy(imageDst.data(), dev_imageDst, imageDst.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

		// cleanup
		cudaFree(dev_imageSrc);
		cudaFree(dev_imageDst);
	}
}


#define swap(A,B) { float temp = A; A = B; B = temp;}

//
// Quick sort
// thanks to https://github.com/khaman1/GPU-QuickSort-Algorithm
//
__device__ void csort(uint8_t ls[], int l, int r) {
	int i, j, k, p, q;
	float v;
	if (r <= l)
		return;
	v = ls[r];
	i = l - 1;
	j = r;
	p = l - 1;
	q = r;
	for (;;) {
		while (ls[++i] < v);
		while (v < ls[--j])
			if (j == l)
				break;
		if (i >= j)
			break;
		swap(ls[i], ls[j]);
		if (ls[i] == v) {
			p++;
			swap(ls[p], ls[i]);
		}
		if (v == ls[j]) {
			q--;
			swap(ls[q], ls[j]);
		}
	}
	swap(ls[i], ls[r]);
	j = i - 1;
	i++;
	for (k = l; k < p; k++, j--)
		swap(ls[k], ls[j]);
	for (k = r - 1; k > q; k--, i++)
		swap(ls[k], ls[i]);

	csort(ls, l, j);
	csort(ls, i, r);
}

