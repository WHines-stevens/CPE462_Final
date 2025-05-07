
#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "filters.h"

int main() {
    cv::Mat original = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Error: Could not load input image.";
        return -1;
    }

    cv::Size psfSize(21, 21);
    double sigma = 5.0;
    cv::Mat psf = createGaussianPSF(psfSize, sigma);

    double noiseStdDev = 10.0;
    cv::Mat degraded = degradeImage(original, psf, noiseStdDev);

    cv::Mat restoredInverse = inverseFilter(degraded, psf);
    double snr = 0.01;
    cv::Mat restoredWiener = wienerFilter(degraded, psf, snr);

    showAndSave("Original Image", original, "original.png");
    showAndSave("Degraded Image", degraded, "degraded.png");
    showAndSave("Restored (Inverse Filter)", restoredInverse, "restored_inverse.png");
    showAndSave("Restored (Wiener Filter)", restoredWiener, "restored_wiener.png");

    return 0;
}
