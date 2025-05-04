#pragma once
#include <opencv2/opencv.hpp>

cv::Mat inverseFilter(const cv::Mat& degraded, const cv::Mat& psf);
cv::Mat wienerFilter(const cv::Mat& degraded, const cv::Mat& psf, double K);
