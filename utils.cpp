#include "utils.h"
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat createGaussianPSF(cv::Size size, double sigma) {
    cv::Mat psf = cv::getGaussianKernel(size.height, sigma, CV_32F) *
                  cv::getGaussianKernel(size.width, sigma, CV_32F).t();
    cv::normalize(psf, psf, 0, 1, cv::NORM_MINMAX);
    return psf;
}

cv::Mat degradeImage(const cv::Mat& image, const cv::Mat& psf, double noiseStdDev) {
    cv::Mat imageF, psfF, blurred;
    image.convertTo(imageF, CV_32F);

    // Pad PSF to same size as image
    cv::Mat psfPadded = cv::Mat::zeros(imageF.size(), CV_32F);
    cv::Rect roi((psfPadded.cols - psf.cols) / 2, (psfPadded.rows - psf.rows) / 2, psf.cols, psf.rows);
    psf.copyTo(psfPadded(roi));

    // Frequency domain convolution (blur)
    cv::Mat imageDFT, psfDFT;
    cv::Mat planes[] = { imageF, cv::Mat::zeros(imageF.size(), CV_32F) };
    cv::merge(planes, 2, imageDFT);
    cv::dft(imageDFT, imageDFT);

    cv::Mat psfPlanes[] = { psfPadded, cv::Mat::zeros(imageF.size(), CV_32F) };
    cv::merge(psfPlanes, 2, psfDFT);
    cv::dft(psfDFT, psfDFT);

    cv::Mat blurredDFT;
    cv::mulSpectrums(imageDFT, psfDFT, blurredDFT, 0);

    cv::idft(blurredDFT, blurred, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(blurred, blurred, 0, 255, cv::NORM_MINMAX);
    blurred.convertTo(blurred, CV_8U);

    // Add Gaussian noise
    cv::Mat noise = cv::Mat(blurred.size(), CV_8U);
    cv::randn(noise, 0, noiseStdDev);
    cv::Mat degraded = blurred + noise;

    return degraded;
}

void showAndSave(const std::string& windowName, const cv::Mat& img, const std::string& filename) {
    cv::imshow(windowName, img);
    cv::imwrite(filename, img);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}
 
