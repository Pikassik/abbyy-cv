#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat VNGdemosaicing(const Mat& input) {
  std::vector<Mat> BGRin(3);
  split(input, BGRin);
  Mat in_sum = BGRin[0] + BGRin[1] + BGRin[2];
  cout << in_sum.rows << " " << in_sum.cols << std::endl;
  Mat output = Mat::zeros(input.size(), CV_8UC3);
  std::vector<Mat> BGRout(3);
  split(output, BGRout);
  std::cout << in_sum.size() << output.size() << std::endl;
  auto get = [&](int i, int j) -> uchar {
    if ((i >= 0) && (i < in_sum.rows) && (j >= 0) && (j < in_sum.cols)) {
      return in_sum.at<uchar>(i, j);
    } else {
      return 0;
    }
  };

  for (int i = 0; i < in_sum.rows; ++i) {
    bool is_green = (i % 2) != 0;

    bool is_blue_above = (i % 2) == 0;
    // else red above
    // (blue above -> red in row) else (red above -> blue in row)

    for (int j = 0; j < in_sum.cols; ++j, is_green = !is_green) {
      double gN = (
              abs(get(i - 1, j) - get(i + 1, j)) +
              abs(get(i - 2, j) - get(i, j)) +
              abs(get(i - 1, j - 1) - get(i + 1, j - 1)) / 2. +
              abs(get(i - 1, j + 1) - get(i + 1, j + 1)) / 2. +
              abs(get(i - 2, j - 1) - get(i, j - 1)) / 2. +
              abs(get(i - 2, j + 1) - get(i, j + 1)) / 2.
              );
      double gE = (
              abs(get(i, j + 1) - get(i, j - 1)) +
              abs(get(i, j + 2) - get(i, j)) +
              abs(get(i - 1, j + 1) - get(i - 1, j - 1)) / 2. +
              abs(get(i + 1, j + 1) - get(i + 1, j - 1)) / 2. +
              abs(get(i - 1, j + 2) - get(i - 1, j)) / 2. +
              abs(get(i + 1, j + 2) - get(i + 1, j)) / 2.
              );
      double gS = (
              abs(get(i + 1, j) - get(i - 1, j)) +
              abs(get(i + 2, j) - get(i, j)) +
              abs(get(i + 1, j + 1) - get(i - 1, j + 1)) / 2. +
              abs(get(i + 1, j - 1) - get(i - 1, j - 1)) / 2. +
              abs(get(i + 2, j + 1) - get(i, j + 1)) / 2. +
              abs(get(i + 2, j - 1) - get(i, j - 1)) / 2.
              );
      double gW = (
              abs(get(i, j - 1) - get(i, j + 1)) +
              abs(get(i, j - 2) - get(i, j)) +
              abs(get(i + 1, j - 1) - get(i + 1, j + 1)) / 2. +
              abs(get(i - 1, j - 1) - get(i - 1, j + 1)) / 2. +
              abs(get(i + 1, j - 2) - get(i + 1, j)) / 2. +
              abs(get(i - 1, j - 2) - get(i - 1, j)) / 2.
              );
      double gNE = (
              abs(get(i - 1, j + 1) - get(i + 1, j - 1)) +
              abs(get(i - 2, j + 2) - get(i, j)) +
              abs(get(i - 1, j) - get(i, j - 1)) / 2. +
              abs(get(i, j + 1) - get(i + 1, j)) / 2. +
              abs(get(i - 2, j + 1) - get(i - 1, j)) / 2. +
              abs(get(i - 1, j + 2) - get(i, j + 1)) / 2.
              );
      double gSE = (
              abs(get(i + 1, j + 1) - get(i - 1, j - 1)) +
              abs(get(i + 2, j + 2) - get(i, j)) +
              abs(get(i, j - 1) - get(i + 1, j)) / 2. +
              abs(get(i - 1, j) - get(i, j + 1)) / 2. +
              abs(get(i + 1, j + 2) - get(i, j + 1)) / 2. +
              abs(get(i + 2, j + 1) - get(i + 1, j)) / 2.
              );
      double gNW = (
              abs(get(i - 1, j - 1) - get(i + 1, j + 1)) +
              abs(get(i - 2, j - 2) - get(i, j)) +
              abs(get(i, j - 1) - get(i + 1, j)) / 2. +
              abs(get(i - 1, j) - get(i, j + 1)) / 2. +
              abs(get(i - 1, j - 2) - get(i, j - 1)) / 2. +
              abs(get(i - 2, j - 1) - get(i - 1, j)) / 2.
              );
      double gSW = (
              abs(get(i + 1, j - 1) - get(i - 1, j + 1)) +
              abs(get(i + 2, j - 2) - get(i, j)) +
              abs(get(i + 1, j) - get(i, j + 1)) / 2. +
              abs(get(i, j - 1) - get(i - 1, j)) / 2. +
              abs(get(i + 2, j - 1) - get(i + 1, j)) / 2. +
              abs(get(i + 1, j - 2) - get(i, j - 1)) / 2.
              );

      auto grads = std::array{gN, gE, gS, gW, gNE, gSE, gNW, gSW};
      double gMin = *std::min(grads.begin(), grads.end());
      double gMax = *std::max(grads.begin(), grads.end());
      constexpr double k1 = 1.5, k2 = 0.5;
      double T = k1 * gMin + k2 * (gMax - gMin); // or gMax + gMin ? same as in opencv
      int grads_cnt = std::max(
              count_if(grads.begin(), grads.end(), [&](double grad) {return grad <= T;}),
              1l
      );

      double Gsum = 0, Bsum = 0, Rsum = 0;
      auto shrink = [](int x) {return std::max(std::min(x, 255), 0);};
      if (is_green) {
        // blue above and red above

        // red above case
        if (gN <= T) {
          Gsum += (get(i - 2, j) + get(i, j)) / 2.;
          Bsum += (get(i, j - 1) + get(i, j + 1) +
                   get(i - 2, j - 1) + get(i - 2, j + 1)) / 4.;
          Rsum += get(i - 1, j);
        }
        if (gE <= T) {
          Gsum += (get(i, j + 2) + get(i, j)) / 2.;
          Bsum += get(i, j + 1);
          Rsum += (get(i - 1, j + 2) + get(i - 1, j) +
                   get(i + 1, j + 2) + get(i + 1, j)) / 4.;
        }
        if (gS <= T) {
          Gsum += (get(i + 2, j) + get(i, j)) / 2.;
          Bsum += (get(i, j - 1) + get(i, j + 1) +
                   get(i + 2, j - 1) + get(i + 2, j + 1)) / 4.;
          Rsum += get(i + 1, j);
        }
        if (gW <= T) {
          Gsum += (get(i, j - 2) + get(i, j)) / 2.;
          Bsum += get(i, j - 1);
          Rsum += (get(i - 1, j - 2) + get(i - 1, j) +
                   get(i + 1, j - 2) + get(i + 1, j)) / 4.;
        }
        if (gNE <= T) {
          Gsum += get(i - 1, j + 1);
          Bsum += (get(i - 2, j + 1) + get(i, j + 1)) / 2.;
          Rsum += (get(i - 1, j) + get(i - 1, j + 2)) / 2.;
        }
        if (gSE <= T) {
          Gsum += get(i + 1, j + 1);
          Bsum += (get(i, j + 1) + get(i + 2, j + 1)) / 2.;
          Rsum += (get(i + 1, j) + get(i + 1, j + 2)) / 2.;
        }
        if (gNW <= T) {
          Gsum += get(i - 1, j - 1);
          Bsum += (get(i, j - 1) + get(i - 2, j - 1)) / 2.;
          Rsum += (get(i - 1, j - 2) + get(i - 1, j)) / 2.;
        }
        if (gSW <= T) {
          Gsum += get(i + 1, j - 1);
          Bsum += (get(i, j - 1) + get(i + 2, j - 1)) / 2.;
          Rsum += (get(i + 1, j - 2) + get(i + 1, j)) / 2.;
        }

        if (is_blue_above) {
          std::swap(Bsum, Rsum);
        }

        int BGdiff = (int)((Bsum - Gsum) / grads_cnt);
        int RGdiff = (int)((Rsum - Gsum) / grads_cnt);
        int G = BGRin[1].at<uchar>(i, j);
        BGRout[1].at<uchar>(i, j) = G;
        BGRout[0].at<uchar>(i, j) = shrink(G + BGdiff);
        BGRout[2].at<uchar>(i, j) = shrink(G + RGdiff);

      } else {
        // red center case
        if (gN <= T) {
          Gsum += get(i - 1, j);
          Bsum += (get(i - 1, j - 1) + get(i - 1, j + 1)) / 2.;
          Rsum += (get(i, j) + get(i - 2, j)) / 2.;
        }
        if (gE <= T) {
          Gsum += get(i, j + 1);
          Bsum += (get(i - 1, j + 1) + get(i + 1, j + 1)) / 2.;
          Rsum += (get(i, j + 2) + get(i, j)) / 2.;
        }
        if (gS <= T) {
          Gsum += get(i + 1, j);
          Bsum += (get(i + 1, j + 1) + get(i + 1, j - 1)) / 2.;
          Rsum += (get(i + 2, j) + get(i, j)) / 2.;
        }
        if (gW <= T) {
          Gsum += get(i, j - 1);
          Bsum += (get(i - 1, j - 1) + get(i + 1, j - 1)) / 2.;
          Rsum += (get(i, j) + get(i, j - 2)) / 2.;
        }
        if (gNE <= T) {
          Gsum += (get(i - 1, j) + get(i, j + 1) +
                   get(i - 2, j + 1) + get(i - 1, j + 2)) / 4.;
          Bsum += get(i - 1, j + 1);
          Rsum += (get(i, j) + get(i - 2, j + 2)) / 2.;
        }
        if (gSE <= T) {
          Gsum += (get(i, j + 1) + get(i + 1, j) +
                   get(i + 1, j + 2) + get(i + 2, j + 1)) / 4.;
          Bsum += get(i + 1, j + 1);
          Rsum += (get(i, j) + get(i + 2, j + 2)) / 2.;
        }
        if (gNW <= T) {
          Gsum += (get(i - 2, j - 1) + get(i - 1, j - 2) +
                   get(i - 1, j) + get(i, j - 1)) / 4.;
          Bsum += get(i - 1, j - 1);
          Rsum += (get(i - 2, j - 2) + get(i, j)) / 2.;
        }
        if (gSW <= T) {
          Gsum += (get(i, j - 1) + get(i + 1, j - 2) +
                   get(i + 1, j) + get(i + 2, j - 1)) / 4.;
          Bsum += get(i + 1, j - 1);
          Rsum += (get(i, j) + get(i + 2, j - 2)) / 2.;
        }

        if (is_blue_above) {
          int GRdiff = (int)((Gsum - Rsum) / grads_cnt);
          int BRdiff = (int)((Bsum - Rsum) / grads_cnt);
          int R = BGRin[2].at<uchar>(i, j);
          BGRout[2].at<uchar>(i, j) = R;
          BGRout[0].at<uchar>(i, j) = shrink(R + BRdiff);
          BGRout[1].at<uchar>(i, j) = shrink(R + GRdiff);

        } else {
          // not blue above -> red above -> blue in row
          std::swap(Bsum, Rsum);
          int GBdiff = (int)((Gsum - Bsum) / grads_cnt);
          int RBdiff = (int)((Rsum - Bsum) / grads_cnt);
          int B = BGRin[0].at<uchar>(i, j);
          BGRout[0].at<uchar>(i, j) = B;
          BGRout[1].at<uchar>(i, j) = shrink(B + GBdiff);
          BGRout[2].at<uchar>(i, j) = shrink(B + RBdiff);
        }
      }
    }
  }
  merge(BGRout, output);
  return output;
}

int main() {
  using namespace cv;
  using namespace std::chrono;
  Mat image = imread("../zadanie1/RGB_CFA.bmp");
  if (image.empty())
  {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }
//  std::vector<Mat> BGR;
//  split(image, BGR); // merge to reverse split
//  Mat output = Mat::zeros(image.size(), CV_8UC3);
  auto start = steady_clock::now();
  Mat output_image = VNGdemosaicing(image);
  auto end = steady_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  std::cout << "transformed in " << duration.count() << " ms" << std::endl;
  imwrite("output.bmp", output_image);
//  String windowName = "VNG"; //Name of the window
//
//  namedWindow(windowName); // Create a window
//
//  imshow(windowName, output_image); // Show our image inside the created window.
//
//  waitKey(0); // Wait for any keystroke in the window
//
//  destroyWindow(windowName); //destroy the created window
//  std::cout << "Hello, World!" << std::endl;
//  std::cout << image.size() << std::endl;
//  std::cout << image.channels() << std::endl;
//  std::cout << (int)BGR[0].at<uint8_t>(1, 1) << std::endl;
  return 0;
}
