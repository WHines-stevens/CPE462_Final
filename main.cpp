#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
#include "utils.h"

int main() {
    // Load original and degraded image (grayscale)
    cv::Mat degraded = cv::imread("images/degraded.png", cv::IMREAD_GRAYSCALE);
    cv::Mat original = cv::imread("images/original.png", cv::IMREAD_GRAYSCALE);

    if (degraded.empty() || original.empty()) {
        std::cerr << "Error: Could not load images.\n";
        return -1;
    }

    // Create horizontal motion blur kernel (e.g., length 15)
    int blurLength = 15;
    cv::Mat psf = createMotionBlurKernel(blurLength, degraded.size());

    // Apply inverse filtering
    cv::Mat restored_inv = inverseFilter(degraded, psf);
    cv::imwrite("results/restored_inverse.png", restored_inv);

    // Apply Wiener filtering
    double K = 0.01; // noise-to-signal power ratio (assumed)
    cv::Mat restored_wiener = wienerFilter(degraded, psf, K);
    cv::imwrite("results/restored_wiener.png", restored_wiener);

    // Compute and print PSNR
    double psnr_inv = calculatePSNR(original, restored_inv);
    double psnr_wiener = calculatePSNR(original, restored_wiener);

    std::cout << "PSNR (Inverse Filter): " << psnr_inv << " dB\n";
    std::cout << "PSNR (Wiener Filter): " << psnr_wiener << " dB\n";

    return 0;
}

