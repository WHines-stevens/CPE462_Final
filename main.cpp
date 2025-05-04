#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "filters.h"

int main() {
    // Load input grayscale image
    cv::Mat original = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Error: Could not load input image.\n";
        return -1;
    }

    // Step 1: Create Gaussian PSF (blur kernel)
    cv::Size psfSize(21, 21);     // Size of the blur kernel
    double sigma = 5.0;           // Standard deviation for Gaussian blur
    cv::Mat psf = createGaussianPSF(psfSize, sigma);

    // Step 2: Degrade image (blur + noise)
    double noiseStdDev = 10.0;    // Standard deviation of Gaussian noise
    cv::Mat degraded = degradeImage(original, psf, noiseStdDev);

    // Step 3: Apply inverse filter
    cv::Mat restoredInverse = inverseFilter(degraded, psf);

    // Step 4: Apply Wiener filter
    double snr = 0.01;            // Lower value = more noise assumed
    cv::Mat restoredWiener = wienerFilter(degraded, psf, snr);

    // Step 5: Display and save results
    showAndSave("Original Image", original, "original.png");
    showAndSave("Degraded Image", degraded, "degraded.png");
    showAndSave("Restored (Inverse Filter)", restoredInverse, "restored_inverse.png");
    showAndSave("Restored (Wiener Filter)", restoredWiener, "restored_wiener.png");

    return 0;
}
