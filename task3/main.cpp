#include <iostream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;


int TrueLog2(int x) {
  assert(x >= 1);
  int k = 0;
  int y = 1;
  while (y < x) {
    y *= 2;
    k += 1;
  }
  return k;
}

Mat FastHoughTransform(const Mat& input) {
  Mat grayscale_input;
  cv::cvtColor(input, grayscale_input, cv::COLOR_BGR2GRAY);
  int k = std::max(TrueLog2(grayscale_input.rows), TrueLog2(grayscale_input.cols));
  int pad_bottom = (1 << k) - grayscale_input.rows;
  int pad_right = (1 << k) - grayscale_input.cols;
  Mat grayscale_input_pad;
  cv::copyMakeBorder(grayscale_input, grayscale_input_pad, 0, pad_bottom, 0, pad_right,  cv::BORDER_CONSTANT);
  Mat im2 = grayscale_input_pad.clone();
  cv::transpose(im2, im2);

  Mat im3 = grayscale_input_pad.clone();
  cv::transpose(im3, im3);
  cv::flip(im3, im3, 1);

  std::vector<Mat> ims{im2, im3};

  std::vector<Mat> Rs;

  int f = 1;
  for (auto& im: ims) {
    Mat R1;
    cv::copyMakeBorder(im, R1, 0, 0, 0, im.cols, cv::BORDER_CONSTANT);
    R1.convertTo(R1, CV_32F);

    Mat R2 = Mat::zeros(R1.rows, R1.cols, R1.type());


    int N = R1.rows;

    for (int i = 1; i <= k; ++i) {
      for (int a = 0; a < (1 << i); ++a) {
        for (int y = 0; y < N; y += (1 << i)) {
          for (int x = 0; x < 2 * N; ++x) {
            int adiv2l = a / 2;
            int adiv2u = adiv2l + a % 2;
            int x_left = x - adiv2u;
            if (x_left < 0) {
              x_left += 2 * N;
            }
            R2.at<float>(y + a, x) =
                    R1.at<float>(y + adiv2l, x) +
                    R1.at<float>(y + (1 << (i - 1)) + adiv2l, x_left);
          }
        }
      }
      cv::swap(R1, R2);
    }

    Mat R = Mat::zeros(450, 1000, R1.type());
    for (int d = 0; d < R.cols; ++d) {
      for (int t = 0; t < R.rows; ++t) {
        double trad = t / 4. / R.rows;
        double dn = (sqrt(2) * N) / R.cols * d;
        if (f == 1) {
          dn = (int)(dn - N * (sqrt(2) - 1) + 2 * N) % (2 * N);
        }
        int dp = (int) (dn / std::cos(trad));
        R.at<float>(t, d) = R1.at<float>((int) ((N - 1) * std::tan(trad)), dp);
      }
    }

    Rs.push_back(R);
    f += 1;
  }

  Mat res = Mat::zeros(Rs[0].rows * 2, Rs[0].cols, CV_32F);

  cv::flip(Rs[0], Rs[0], 0);
  cv::flip(Rs[0], Rs[0], 1);
  Rs[0].copyTo(res(Rect(0, 0, Rs[0].cols, Rs[0].rows)));
  Rs[1].copyTo(res(Rect(0, Rs[0].rows * 1, Rs[0].cols, Rs[0].rows)));
  double minVal, maxVal;
  Point minLoc, maxLoc;
  minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

  res = (res - minVal) / (maxVal - minVal) * 255.;

  return res;
}

double GetRotationAngle(const Mat& hough) {
  double varmax = -1;
  int argmax = 0;
  for (int i = 0; i < hough.rows; ++i) {
    double var = 0.;
    double mean = 0.;
    for (int j = 0; j < hough.cols; ++j) {
      double x = hough.at<uchar>(i, j);
      mean += x / hough.cols;
    }

    for (int j = 0; j < hough.cols; ++j) {
      double x = hough.at<float>(i, j);
      var += (x - mean) * (x - mean) / hough.cols;
    }
    if (var > varmax) {
      varmax = var;
      argmax = i;
    }
  }


  return 45 - static_cast<double>(argmax) / hough.rows * 90;
}

int main(int argc, char** argv) {
  using namespace cv;
  using namespace std::chrono;

  if (argc < 3) {
    std::cout << "usage: <bin> <input> <output>" << std::endl;
    return -1;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  Mat image = imread(input);
  if (image.empty())
  {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  auto t0 = steady_clock::now();
  Mat output_image = FastHoughTransform(image);
  auto t1 = steady_clock::now();
  double angle = GetRotationAngle(output_image);

  output_image.convertTo(output_image, CV_8U);

  std::cout << duration_cast<milliseconds>(t1 - t0).count() << std::endl;
  std::cout << angle << std::endl;

  imwrite(output, output_image);
  return 0;
}
