import cv2
import time
import math
import numpy as np


def medianFilter(imageSrc: np.array, imageDst: np.array, kernelSize: int):

    medianIndex = int((kernelSize * kernelSize + 1) / 2)
    kernelRadius = math.floor(kernelSize / 2)

    [rows, cols, channels] = imageSrc.shape
    pixelValues = np.zeros((kernelSize * kernelSize), dtype=np.uint8)

    for row in range(kernelRadius, rows - kernelRadius):
        for col in range(kernelRadius, cols - kernelRadius):
            for channel in range(0, channels):

                i = 0
                for kernel_row in range(-kernelRadius, kernelRadius + 1):
                    for kernel_col in range(-kernelRadius, kernelRadius + 1):
                        pixelValues[i] = imageSrc[row +
                                                  kernel_row, col + kernel_col, channel]
                        i += 1

                np.sort(pixelValues)
                imageDst[row, col, channel] = pixelValues[medianIndex]


if __name__ == "__main__":

    imageSrc = cv2.imread("images/Chaplin_noisy.png")
    imageSrc = imageSrc.astype(np.uint8)
    # imageSrc /= 255
    imageDst = np.zeros(imageSrc.shape, dtype=np.float)
    kernelSize = 3

    t = time.time()
    medianFilter(imageSrc, imageDst, kernelSize)
    print((time.time() - t) * 1000000, " milliseconds")

    cv2.imshow("src", imageSrc)
    cv2.imshow("dst", imageDst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
