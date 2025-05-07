#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "filters.h"

int main() {
    // Load the input image
    cv::Mat original = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Error: Could not load input image." << std::endl;
        return -1;
    }

    // Based on the images you shared, we need a smaller PSF and more aggressive filters
    cv::Size psfSize(9, 9);  // Smaller PSF for high-frequency detail preservation
    double sigma = 1.5;      // Lower sigma for less severe blur
    cv::Mat psf = createGaussianPSF(psfSize, sigma);

    // If the image is already degraded (like the ones you shared), use it directly
    cv::Mat degraded;
    
    if (original.rows > 100 && cv::countNonZero(original) > 0) {
        // Check if the image looks like noise (has high standard deviation relative to mean)
        cv::Scalar mean, stddev;
        cv::meanStdDev(original, mean, stddev);
        double noiseRatio = stddev[0] / (mean[0] + 0.01);  // Avoid division by zero
        
        if (noiseRatio > 0.5) {
            // If input image already appears to be degraded/noisy
            std::cout << "Input appears to be a degraded image. Using it directly." << std::endl;
            degraded = original.clone();
        } else {
            // Create a simulated degraded image with low noise
            double noiseStdDev = 2.0;  // Very low noise to make restoration easier
            degraded = degradeImage(original, psf, noiseStdDev);
        }
    } else {
        std::cerr << "Warning: Input image issues detected." << std::endl;
        degraded = original.clone();  // Fallback
    }

    // Apply restoration filters with more aggressive parameters for noisy input
    cv::Mat restoredInverse = inverseFilter(degraded, psf);
    
    // Higher K for more noise suppression in the Wiener filter
    double wienerK = 0.05;  // Increased from 0.01 to handle more noise
    cv::Mat restoredWiener = wienerFilter(degraded, psf, wienerK);

    // Display and save results
    std::cout << "Processing complete. Saving images..." << std::endl;
    
    showAndSave("Original Image", original, "original.png");
    showAndSave("Degraded Image", degraded, "degraded.png");
    showAndSave("Restored (Inverse Filter)", restoredInverse, "restored_inverse.png");
    showAndSave("Restored (Wiener Filter)", restoredWiener, "restored_wiener.png");

    std::cout << "Done! Images saved to disk." << std::endl;
    return 0;
}
