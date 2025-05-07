
#include "filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>

static cv::Mat padAndConvert(const cv::Mat& input, cv::Size size) {
    cv::Mat padded;
    input.convertTo(padded, CV_32F);
    if (padded.size() != size)
        cv::copyMakeBorder(padded, padded, 0, size.height - padded.rows, 0, size.width - padded.cols, cv::BORDER_CONSTANT, 0);
    return padded;
}

static void computeDFT(const cv::Mat& input, cv::Mat& complexI) {
    cv::Mat planes[] = { input, cv::Mat::zeros(input.size(), CV_32F) };
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
}

static cv::Mat computeIDFT(const cv::Mat& complexI) {
    cv::Mat invDFT;
    cv::idft(complexI, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    invDFT.convertTo(invDFT, CV_8U);
    return invDFT;
}

cv::Mat inverseFilter(const cv::Mat& degraded, const cv::Mat& psf) {
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(degraded.cols + psf.cols - 1);
    dftSize.height = cv::getOptimalDFTSize(degraded.rows + psf.rows - 1);

    cv::Mat degradedF = padAndConvert(degraded, dftSize);
    cv::Mat psfF = padAndConvert(psf, dftSize);

    cv::Mat G, H;
    computeDFT(degradedF, G);
    computeDFT(psfF, H);

    cv::Mat H_planes[2];
    cv::split(H, H_planes);
    cv::Mat H_mag;
    cv::magnitude(H_planes[0], H_planes[1], H_mag);

    for (int y = 0; y < H_mag.rows; y++)
        for (int x = 0; x < H_mag.cols; x++)
            if (H_mag.at<float>(y, x) < 1e-6f) {
                H_planes[0].at<float>(y, x) = 1e-6f;
                H_planes[1].at<float>(y, x) = 0.0f;
            }

    cv::merge(H_planes, 2, H);
    cv::Mat F;
    cv::divide(G, H, F);
    cv::Mat restored = computeIDFT(F);
    return restored(cv::Rect(0, 0, degraded.cols, degraded.rows));
}

cv::Mat wienerFilter(const cv::Mat& degraded, const cv::Mat& psf, double K) {
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(degraded.cols + psf.cols - 1);
    dftSize.height = cv::getOptimalDFTSize(degraded.rows + psf.rows - 1);

    cv::Mat degradedF = padAndConvert(degraded, dftSize);
    cv::Mat psfF = padAndConvert(psf, dftSize);

    cv::Mat G, H;
    computeDFT(degradedF, G);
    computeDFT(psfF, H);

    cv::Mat H_conj;
    cv::mulSpectrums(H, H, H_conj, 0, true);
    cv::Mat denom = H_conj + K;

    cv::Mat Wiener;
    cv::mulSpectrums(G, H, Wiener, 0, true);
    cv::divide(Wiener, denom, Wiener);

    cv::Mat restored = computeIDFT(Wiener);
    return restored(cv::Rect(0, 0, degraded.cols, degraded.rows));
}
