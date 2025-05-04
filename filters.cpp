#include "filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Helper: Convert to float and expand to optimal DFT size
static cv::Mat padAndConvert(const cv::Mat& input, cv::Size size) {
    cv::Mat padded;
    input.convertTo(padded, CV_32F);
    if (padded.size() != size)
        cv::copyMakeBorder(padded, padded, 0, size.height - input.rows, 0, size.width - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return padded;
}

// Helper: Compute DFT of an image
static void computeDFT(const cv::Mat& input, cv::Mat& complexI) {
    cv::Mat planes[] = { input, cv::Mat::zeros(input.size(), CV_32F) };
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
}

// Helper: Inverse DFT and extract real part
static cv::Mat computeIDFT(const cv::Mat& complexI) {
    cv::Mat invDFT, planes[2];
    cv::idft(complexI, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    invDFT.convertTo(invDFT, CV_8U);
    return invDFT;
}

cv::Mat inverseFilter(const cv::Mat& degraded, const cv::Mat& psf) {
    cv::Size dftSize = degraded.size();
    cv::Mat degradedF = padAndConvert(degraded, dftSize);
    cv::Mat psfF = padAndConvert(psf, dftSize);

    cv::Mat G, H;
    computeDFT(degradedF, G);
    computeDFT(psfF, H);

    cv::Mat H_mag;
    cv::magnitude(H.reshape(2).col(0), H.reshape(2).col(1), H_mag);

    // Avoid division by near-zero
    cv::Mat mask = H_mag < 1e-6;
    H.setTo(cv::Scalar(1e-6, 0), mask);

    cv::Mat F;
    cv::divide(G, H, F); // F = G / H

    return computeIDFT(F);
}

cv::Mat wienerFilter(const cv::Mat& degraded, const cv::Mat& psf, double K) {
    cv::Size dftSize = degraded.size();
    cv::Mat degradedF = padAndConvert(degraded, dftSize);
    cv::Mat psfF = padAndConvert(psf, dftSize);

    cv::Mat G, H;
    computeDFT(degradedF, G);
    computeDFT(psfF, H);

    cv::Mat H_conj;
    cv::mulSpectrums(H, H, H_conj, 0, true); // |H|^2

    cv::Mat denom;
    denom = H_conj + K;

    cv::Mat Wiener;
    cv::mulSpectrums(G, H, Wiener, 0, true); // G * H*
    cv::divide(Wiener, denom, Wiener);

    return computeIDFT(Wiener);
}
