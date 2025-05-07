#include "filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Utility function to prepare image for DFT processing
static cv::Mat padAndConvert(const cv::Mat& input, cv::Size size) {
    cv::Mat padded;
    input.convertTo(padded, CV_32F);
    if (padded.size() != size) {
        cv::copyMakeBorder(padded, padded, 0, size.height - padded.rows, 0, 
                          size.width - padded.cols, cv::BORDER_CONSTANT, 0);
    }
    return padded;
}

// Compute the Discrete Fourier Transform (DFT)
static void computeDFT(const cv::Mat& input, cv::Mat& complexI) {
    cv::Mat planes[] = { input, cv::Mat::zeros(input.size(), CV_32F) };
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
}

// Compute the Inverse Discrete Fourier Transform (IDFT)
static cv::Mat computeIDFT(const cv::Mat& complexI) {
    cv::Mat invDFT;
    cv::idft(complexI, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    // Apply contrast stretching for better visualization
    double minVal, maxVal;
    cv::minMaxLoc(invDFT, &minVal, &maxVal);
    
    // Handle extreme pixel values - clip to avoid overflow
    cv::Mat clipped;
    double p1 = 0.01, p99 = 0.99; // Percentiles for robust scaling
    
    // Sort the values to find percentiles
    std::vector<float> pixels;
    invDFT.reshape(1, 1).copyTo(pixels);
    std::sort(pixels.begin(), pixels.end());
    
    // Use percentiles to avoid outliers affecting the scaling
    float lowVal = pixels[static_cast<int>(p1 * pixels.size())];
    float highVal = pixels[static_cast<int>(p99 * pixels.size())];
    
    // Clip and scale values between the percentiles
    cv::Mat normalized;
    invDFT.convertTo(normalized, CV_32F);
    normalized = cv::max(cv::min(normalized, highVal), lowVal);
    
    // Final scaling to 8-bit range
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    
    // Apply a mild sharpening filter to enhance details
    cv::Mat sharpened;
    cv::GaussianBlur(normalized, sharpened, cv::Size(0, 0), 1.0);
    cv::addWeighted(normalized, 1.5, sharpened, -0.5, 0, sharpened);
    
    return sharpened;
}

// Center the PSF for proper convolution
static cv::Mat centerPSF(const cv::Mat& psf, cv::Size dftSize) {
    cv::Mat centeredPSF = cv::Mat::zeros(dftSize, CV_32F);
    
    // Copy PSF to the center of the frequency domain
    int dx = (dftSize.width - psf.cols) / 2;
    int dy = (dftSize.height - psf.rows) / 2;
    
    cv::Rect roi(dx, dy, psf.cols, psf.rows);
    psf.copyTo(centeredPSF(roi));
    
    // Circularly shift PSF for proper convolution
    cv::Mat tmp = centeredPSF.clone();
    int cx = centeredPSF.cols / 2;
    int cy = centeredPSF.rows / 2;
    
    // Top-left -> Bottom-right
    cv::Mat q0(tmp, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(tmp, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(tmp, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(tmp, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat d0(centeredPSF, cv::Rect(cx, cy, cx, cy));
    cv::Mat d1(centeredPSF, cv::Rect(0, cy, cx, cy));
    cv::Mat d2(centeredPSF, cv::Rect(cx, 0, cx, cy));
    cv::Mat d3(centeredPSF, cv::Rect(0, 0, cx, cy));
    
    q0.copyTo(d0);
    q1.copyTo(d1);
    q2.copyTo(d2);
    q3.copyTo(d3);
    
    return centeredPSF;
}

cv::Mat inverseFilter(const cv::Mat& degraded, const cv::Mat& psf) {
    // Ensure working with grayscale image
    cv::Mat degradedGray = degraded;
    if (degraded.channels() > 1) {
        cv::cvtColor(degraded, degradedGray, cv::COLOR_BGR2GRAY);
    }
    
    // Calculate optimal DFT size
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(degradedGray.cols);
    dftSize.height = cv::getOptimalDFTSize(degradedGray.rows);
    
    // Prepare images for DFT
    cv::Mat degradedF = padAndConvert(degradedGray, dftSize);
    cv::Mat psfCentered = centerPSF(psf, dftSize);
    cv::Mat psfF = padAndConvert(psfCentered, dftSize);
    
    // Compute DFTs
    cv::Mat G, H;
    computeDFT(degradedF, G);
    computeDFT(psfF, H);
    
    // Implement regularized inverse filter: F = G/(H + ε) where ε is a small constant
    cv::Mat F = cv::Mat(G.size(), G.type());
    
    // Split complex matrices to real and imaginary parts
    cv::Mat planes_G[2], planes_H[2], planes_F[2];
    cv::split(G, planes_G);
    cv::split(H, planes_H);
    
    // Initialize result planes
    planes_F[0] = cv::Mat::zeros(H.rows, H.cols, CV_32F);
    planes_F[1] = cv::Mat::zeros(H.rows, H.cols, CV_32F);
    
    // Much higher threshold to suppress noise in the inverse filter
    const float threshold = 0.1f;  // Increased threshold to suppress noise
    
    // Calculate maximum magnitude of H for adaptive thresholding
    cv::Mat H_mag;
    cv::magnitude(planes_H[0], planes_H[1], H_mag);
    double maxH;
    cv::minMaxLoc(H_mag, nullptr, &maxH);
    const float adaptiveThreshold = threshold * maxH;
    
    // Apply an edge-preserving spectral filter to handle high-frequency components better
    for (int y = 0; y < H.rows; y++) {
        for (int x = 0; x < H.cols; x++) {
            float re_H = planes_H[0].at<float>(y, x);
            float im_H = planes_H[1].at<float>(y, x);
            float mag_H_squared = re_H * re_H + im_H * im_H;
            
            // Get distance from center (frequency measure)
            float dx = x - H.cols/2;
            float dy = y - H.rows/2;
            float distance = std::sqrt(dx*dx + dy*dy) / std::sqrt((H.cols/2)*(H.cols/2) + (H.rows/2)*(H.rows/2));
            
            // Apply frequency-dependent regularization - more regularization for high frequencies
            float dynamic_threshold = adaptiveThreshold * (1.0f + distance);
            
            if (mag_H_squared > dynamic_threshold) {
                float re_G = planes_G[0].at<float>(y, x);
                float im_G = planes_G[1].at<float>(y, x);
                
                // Add regularization term to denominator
                float regularization = dynamic_threshold * 0.01f;
                float denom = mag_H_squared + regularization;
                
                // Complex division with regularization
                planes_F[0].at<float>(y, x) = (re_G * re_H + im_G * im_H) / denom;
                planes_F[1].at<float>(y, x) = (im_G * re_H - re_G * im_H) / denom;
            } else {
                // Zero out frequencies where PSF is too small (avoid noise amplification)
                planes_F[0].at<float>(y, x) = 0;
                planes_F[1].at<float>(y, x) = 0;
            }
        }
    }
    
    cv::merge(planes_F, 2, F);
    cv::Mat restored = computeIDFT(F);
    
    // Enhance contrast of the result
    cv::normalize(restored, restored, 0, 255, cv::NORM_MINMAX);
    
    // Crop to original size
    return restored(cv::Rect(0, 0, degradedGray.cols, degradedGray.rows));
}

cv::Mat wienerFilter(const cv::Mat& degraded, const cv::Mat& psf, double K) {
    // Ensure working with grayscale image
    cv::Mat degradedGray = degraded;
    if (degraded.channels() > 1) {
        cv::cvtColor(degraded, degradedGray, cv::COLOR_BGR2GRAY);
    }
    
    // Calculate optimal DFT size
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(degradedGray.cols);
    dftSize.height = cv::getOptimalDFTSize(degradedGray.rows);
    
    // Prepare images for DFT
    cv::Mat degradedF = padAndConvert(degradedGray, dftSize);
    cv::Mat psfF = padAndConvert(psf, dftSize);
    
    // Center the PSF manually to avoid potential issues
    cv::Point2d center(psfF.cols/2, psfF.rows/2);
    double sum = cv::sum(psfF)[0];
    psfF /= sum;  // Normalize
    
    cv::Mat tmp = psfF.clone();
    int cx = psfF.cols / 2;
    int cy = psfF.rows / 2;
    
    cv::Mat q0(tmp, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(tmp, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(tmp, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(tmp, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat d0(psfF, cv::Rect(cx, cy, cx, cy));
    cv::Mat d1(psfF, cv::Rect(0, cy, cx, cy));
    cv::Mat d2(psfF, cv::Rect(cx, 0, cx, cy));
    cv::Mat d3(psfF, cv::Rect(0, 0, cx, cy));
    
    q0.copyTo(d0);
    q1.copyTo(d1);
    q2.copyTo(d2);
    q3.copyTo(d3);
    
    // Create the complex matrices directly
    cv::Mat degradedComplex, psfComplex;
    
    // Convert to complex format
    cv::Mat planes1[] = {degradedF, cv::Mat::zeros(degradedF.size(), CV_32F)};
    cv::Mat planes2[] = {psfF, cv::Mat::zeros(psfF.size(), CV_32F)};
    
    cv::merge(planes1, 2, degradedComplex);
    cv::merge(planes2, 2, psfComplex);
    
    // Perform DFT
    cv::dft(degradedComplex, degradedComplex);
    cv::dft(psfComplex, psfComplex);
    
    // Split into real and imaginary parts
    cv::Mat planesG[2], planesH[2];
    cv::split(degradedComplex, planesG);
    cv::split(psfComplex, planesH);
    
    // Create output complex array
    cv::Mat planesOut[2];
    planesOut[0] = cv::Mat::zeros(planesG[0].size(), CV_32F);
    planesOut[1] = cv::Mat::zeros(planesG[1].size(), CV_32F);
    
    // Apply Wiener filter: H* / (|H|² + K) * G
    for (int i = 0; i < planesH[0].rows; i++) {
        for (int j = 0; j < planesH[0].cols; j++) {
            // Get values from matrices
            float Gr = planesG[0].at<float>(i, j);
            float Gi = planesG[1].at<float>(i, j);
            float Hr = planesH[0].at<float>(i, j);
            float Hi = planesH[1].at<float>(i, j);
            
            // Calculate |H|²
            float H_sqr_mag = Hr*Hr + Hi*Hi;
            
            // Distance from center for frequency-dependent K
            float row_freq = (float)i - planesH[0].rows/2.0f;
            float col_freq = (float)j - planesH[0].cols/2.0f;
            float dist = sqrt(row_freq*row_freq + col_freq*col_freq);
            float normalized_dist = dist / sqrt(planesH[0].rows*planesH[0].rows/4.0f + planesH[0].cols*planesH[0].cols/4.0f);
            
            // Lower K for low frequencies, higher for high frequencies
            float k_adjusted = K * (0.1f + 2.0f * normalized_dist);
            
            // Denominator with adjusted K
            float denom = H_sqr_mag + k_adjusted;
            
            // Avoid division by very small values
            if (denom < 0.0001f) denom = 0.0001f;
            
            // Complex division (H* * G) / (|H|² + K)
            // H* is the complex conjugate of H (Hr, -Hi)
            planesOut[0].at<float>(i, j) = (Hr*Gr + Hi*Gi) / denom;
            planesOut[1].at<float>(i, j) = (Hr*Gi - Hi*Gr) / denom;
            
            // Additional high-frequency suppression
            if (normalized_dist > 0.7f) {
                float suppression = std::max(0.0f, 1.0f - (normalized_dist - 0.7f) / 0.3f);
                planesOut[0].at<float>(i, j) *= suppression;
                planesOut[1].at<float>(i, j) *= suppression;
            }
        }
    }
    
    // Merge real and imaginary parts
    cv::Mat complexOutput;
    cv::merge(planesOut, 2, complexOutput);
    
    // Inverse DFT
    cv::Mat outputImage;
    cv::idft(complexOutput, outputImage, cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
    
    // Crop to original size
    outputImage = outputImage(cv::Rect(0, 0, degradedGray.cols, degradedGray.rows));
    
    // Normalize to 0-255 range
    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8U);
    
    // Apply mild noise reduction
    cv::Mat denoisedImage;
    cv::fastNlMeansDenoising(outputImage, denoisedImage, 5, 7, 21);
    
    // Apply subtle sharpening
    cv::Mat blurred, sharpened;
    cv::GaussianBlur(denoisedImage, blurred, cv::Size(0, 0), 1.5);
    cv::addWeighted(denoisedImage, 1.5, blurred, -0.5, 0, sharpened);
    
    return sharpened;
}
