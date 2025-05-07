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
    // Convert to grayscale if needed
    cv::Mat imgGray;
    if (degraded.channels() > 1) {
        cv::cvtColor(degraded, imgGray, cv::COLOR_BGR2GRAY);
    } else {
        imgGray = degraded.clone();
    }
    
    // Convert to floating point
    cv::Mat imgFloat;
    imgGray.convertTo(imgFloat, CV_32F);
    
    // Calculate optimal DFT size
    int m = cv::getOptimalDFTSize(imgFloat.rows);
    int n = cv::getOptimalDFTSize(imgFloat.cols);
    
    // Pad the image to optimal DFT size
    cv::Mat padded;
    cv::copyMakeBorder(imgFloat, padded, 0, m - imgFloat.rows, 0, 
                      n - imgFloat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Prepare the PSF for frequency domain processing
    cv::Mat psfFloat;
    psf.convertTo(psfFloat, CV_32F);
    
    // Create a padded PSF of the same size as the input image
    cv::Mat psfPadded = cv::Mat::zeros(padded.size(), CV_32F);
    
    // Place the PSF in the center of the padded image
    int px = (psfPadded.cols - psfFloat.cols) / 2;
    int py = (psfPadded.rows - psfFloat.rows) / 2;
    cv::Mat psfRoi = psfPadded(cv::Rect(px, py, psfFloat.cols, psfFloat.rows));
    psfFloat.copyTo(psfRoi);
    
    // Create DFT matrices for image and PSF
    cv::Mat complexImg, complexPsf;
    
    // Create complex planes for DFT
    cv::Mat planes1[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat planes2[] = {psfPadded, cv::Mat::zeros(psfPadded.size(), CV_32F)};
    
    // Merge into complex format
    cv::merge(planes1, 2, complexImg);
    cv::merge(planes2, 2, complexPsf);
    
    // Perform forward DFT
    cv::dft(complexImg, complexImg);
    cv::dft(complexPsf, complexPsf);
    
    // Create planes for result
    cv::Mat planesH[2];
    cv::split(complexPsf, planesH);
    
    // Create output planes
    cv::Mat planesG[2];
    cv::split(complexImg, planesG);
    
    // Result planes
    cv::Mat planesResult[2];
    planesResult[0] = cv::Mat::zeros(planesH[0].size(), CV_32F);
    planesResult[1] = cv::Mat::zeros(planesH[1].size(), CV_32F);
    
    // Apply Wiener filter
    for (int i = 0; i < padded.rows; i++) {
        for (int j = 0; j < padded.cols; j++) {
            // Get the value of H
            float Hr = planesH[0].at<float>(i, j);
            float Hi = planesH[1].at<float>(i, j);
            
            // Get the value of G
            float Gr = planesG[0].at<float>(i, j);
            float Gi = planesG[1].at<float>(i, j);
            
            // Calculate |H|²
            float H_mag_square = Hr*Hr + Hi*Hi;
            
            // Distance from center for adaptive K
            float center_y = padded.rows / 2.0f;
            float center_x = padded.cols / 2.0f;
            float dist_y = (i - center_y) / center_y;
            float dist_x = (j - center_x) / center_x;
            float normalized_dist = sqrt(dist_y*dist_y + dist_x*dist_x);
            
            // Adjust K to be frequency-dependent (lower for low frequencies)
            float K_adjusted = K * (0.05f + normalized_dist * 0.8f);
            
            // Calculate denominator with regularization
            float denom = H_mag_square + K_adjusted;
            
            // Ensure we don't divide by zero
            if (denom < 1e-5f) denom = 1e-5f;
            
            // Calculate complex conjugate of H
            float Hc_r = Hr;
            float Hc_i = -Hi;
            
            // Wiener filter formula: G * H* / (|H|² + K)
            planesResult[0].at<float>(i, j) = (Gr * Hc_r - Gi * Hc_i) / denom;
            planesResult[1].at<float>(i, j) = (Gr * Hc_i + Gi * Hc_r) / denom;
        }
    }
    
    // Merge back to complex format
    cv::Mat complexResult;
    cv::merge(planesResult, 2, complexResult);
    
    // Perform inverse DFT
    cv::Mat result;
    cv::idft(complexResult, result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    // Crop the result to original size
    result = result(cv::Rect(0, 0, imgGray.cols, imgGray.rows));
    
    // Normalize result to 0-255 range
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);
    
    // Apply additional noise reduction
    cv::Mat denoised;
    cv::fastNlMeansDenoising(result, denoised, 7.0, 7, 21);
    
    // Apply a bilateral filter for edge-preserving smoothing
    cv::Mat smoothed;
    cv::bilateralFilter(denoised, smoothed, 5, 50, 50);
    
    // Apply a slight contrast enhancement
    double alpha = 1.2; // Contrast control
    int beta = 10;      // Brightness control
    cv::Mat enhanced;
    smoothed.convertTo(enhanced, CV_8U, alpha, beta);
    
    return enhanced;
}
