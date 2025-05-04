#pragma once
#include <opencv2/opencv.hpp>

// Generates a Gaussian PSF (blur kernel)
cv::Mat createGaussianPSF(cv::Size size, double sigma);

// Applies blur and Gaussian noise to degrade an image
cv::Mat degradeImage(const cv::Mat& image, const cv::Mat& psf, double noiseStdDev);

// Utility for displaying and saving an image
void showAndSave(const std::string& windowName, const cv::Mat& img, const std::string& filename);
