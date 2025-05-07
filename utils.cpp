#include "utils.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat createGaussianPSF(cv::Size size, double sigma) {
    // Create Gaussian kernel using OpenCV's built-in function
    cv::Mat psf = cv::getGaussianKernel(size.height, sigma, CV_32F) * 
                  cv::getGaussianKernel(size.width, sigma, CV_32F).t();
    
    // Normalize PSF to sum to 1
    cv::normalize(psf, psf, 0, 1, cv::NORM_MINMAX);
    return psf;
}

cv::Mat degradeImage(const cv::Mat& image, const cv::Mat& psf, double noiseStdDev) {
    // Convert image to floating point format
    cv::Mat imageF;
    image.convertTo(imageF, CV_32F);
    
    // Create a padded PSF the same size as the image for proper convolution
    cv::Mat psfPadded = cv::Mat::zeros(imageF.size(), CV_32F);
    cv::Rect roi((psfPadded.cols - psf.cols) / 2, 
                (psfPadded.rows - psf.rows) / 2, 
                psf.cols, psf.rows);
    psf.copyTo(psfPadded(roi));
    
    // Apply circular shift to PSF for proper convolution
    cv::Mat tmp = psfPadded.clone();
    int cx = psfPadded.cols / 2;
    int cy = psfPadded.rows / 2;
    
    // Rearrange quadrants (top-left with bottom-right, top-right with bottom-left)
    cv::Mat q0(tmp, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(tmp, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(tmp, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(tmp, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat d0(psfPadded, cv::Rect(cx, cy, cx, cy));
    cv::Mat d1(psfPadded, cv::Rect(0, cy, cx, cy));
    cv::Mat d2(psfPadded, cv::Rect(cx, 0, cx, cy));
    cv::Mat d3(psfPadded, cv::Rect(0, 0, cx, cy));
    
    q0.copyTo(d0);
    q1.copyTo(d1);
    q2.copyTo(d2);
    q3.copyTo(d3);
    
    // Perform convolution using DFT
    cv::Mat complexI, complexH;
    
    // Transform image to frequency domain
    cv::Mat planes[] = { imageF, cv::Mat::zeros(imageF.size(), CV_32F) };
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    
    // Transform PSF to frequency domain
    cv::Mat planesPSF[] = { psfPadded, cv::Mat::zeros(psfPadded.size(), CV_32F) };
    cv::merge(planesPSF, 2, complexH);
    cv::dft(complexH, complexH);
    
    // Multiply in frequency domain (equivalent to convolution in spatial domain)
    cv::Mat complexResult;
    cv::mulSpectrums(complexI, complexH, complexResult, 0);
    
    // Transform back to spatial domain
    cv::Mat blurred;
    cv::idft(complexResult, blurred, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    // Convert to 8-bit format
    cv::normalize(blurred, blurred, 0, 255, cv::NORM_MINMAX);
    cv::Mat blurred8U;
    blurred.convertTo(blurred8U, CV_8U);
    
    // Add Gaussian noise
    cv::Mat noise(blurred8U.size(), CV_8U);
    cv::randn(noise, cv::Scalar(0), cv::Scalar(noiseStdDev));
    
    cv::Mat degraded;
    cv::add(blurred8U, noise, degraded);
    
    return degraded;
}

void showAndSave(const std::string& windowName, const cv::Mat& img, const std::string& filename) {
    if (img.empty()) {
        std::cerr << "[!] Warning: '" << windowName << "' image is empty — not saved." << std::endl;
        return;
    }
    
    // Display image (uncomment if you have a GUI environment)
    // cv::imshow(windowName, img);
    
    // Save image to file
    bool success = cv::imwrite(filename, img);
    if (success) {
        std::cout << "Saved: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save: " << filename << std::endl;
    }
}
