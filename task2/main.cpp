#include <iostream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum class MedianFilterType {
  kNaive,
  kHuang,
  kConstant
};

class Hist {
 public:
  Hist(int cnt) : hist_{0}, cnt_(cnt) {}

  void AddHist(const Hist& other) {
    for (int i = 0; i < hist_.size(); ++i) {
      hist_.at(i) += other.hist_.at(i);
    }
  }

  void SubtractHist(const Hist& other) {
    for (int i = 0; i < hist_.size(); ++i) {
      hist_.at(i) -= other.hist_.at(i);
    }
  }

  void AddElement(int x) {
    hist_.at(x) += 1;
  }

  void RemoveElement(int x) {
    hist_.at(x) -= 1;
  }

  int Median() const {
    int sum = 0;
    int i = 0;

    while ((sum < cnt_ / 2) && i < hist_.size()) {
      sum += hist_[i];
      ++i;
    }

    if (sum > cnt_ / 2) {
      return i - 1;
    } else {
      int second_ind =
            std::distance(
                      hist_.begin(),
                      std::find_if(hist_.begin() + i,
                                   hist_.end(),
                                   [](int x) { return x > 0; })
                                   );
      if (cnt_ % 2 == 0) {
        return ((i - 1 + second_ind) / 2);
      } else {
        return second_ind;
      }
    }
  }
 private:
  std::array<int, 256> hist_;
  const int cnt_;
};

int Get(const Mat& mat, int i, int j) {
  auto get_k = [&](int k, int N) {
    return std::min(std::max(k, 0), N - 1);
  };
  i = get_k(i, mat.rows);
  j = get_k(j, mat.cols);
  return mat.at<uchar>(get_k(i, mat.rows), get_k(j, mat.cols));
}

Mat MedianFilterNaive(const Mat& input, int R) {
  Mat output = Mat::zeros(input.size(), input.type());
  std::vector<int> buffer;
  buffer.reserve((2 * R + 1) * (2 * R + 1));
  for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < input.cols; ++j) {
      Hist h((2 * R + 1) * (2 * R + 1));
      // copy window
      for (int w_i = -R; w_i <= R; ++w_i) {
        for (int w_j = -R; w_j <= R; ++w_j) {
          int x = Get(input, i + w_i, j + w_j);
          buffer.push_back(x);
          h.AddElement(x);
        }
      }

      std::sort(buffer.begin(), buffer.end());
      output.at<uchar>(i, j) = buffer[(buffer.size() / 2)];
      buffer.resize(0);
    }
  }

  return output;
}

Mat MedianFilterHuang(const Mat& input, int R) {
  Mat output = Mat::zeros(input.size(), input.type());

  Hist hist0((2 * R + 1) * (2 * R + 1));
  // initialize hist
  for (int w_i = -R - 1; w_i <= R - 1; ++w_i) {
    for (int w_j = -R; w_j <= R; ++w_j) {
      hist0.AddElement(Get(input, w_i, w_j));
    }
  }

  for (int i = 0; i < input.rows; ++i) {
    for (int w_j = -R; w_j <= R; ++w_j) {
      hist0.RemoveElement(Get(input, i - R - 1, w_j));
      hist0.AddElement(Get(input, i + R, w_j));
    }

    Hist hist = hist0;


    for (int j = 0; j < input.cols; ++j) {
      output.at<uchar>(i, j) = hist.Median();

      for (int w_i = -R; w_i <= R; ++w_i) {
        hist.RemoveElement(Get(input, i + w_i, j - R));
        hist.AddElement(Get(input, i + w_i, j + R + 1));
      }
    }
  }

  return output;
}

Mat MedianFilterConstant(const Mat& input, int R) {
  Mat output = Mat::zeros(input.rows, input.cols, input.type());
  std::vector<Hist> hists(R + input.cols + R + 1, (2 * R + 1));

  //initialize hists
  for (int j = -R; j < input.cols + R + 1; ++j) {
    for (int i = -R; i <= R; ++i) {
      hists.at(j + R).AddElement(Get(input, i, j));
    }
  }

  Hist hist0((2 * R + 1) * (2 * R + 1));
  // initialize hist
  for (int w_i = -R - 1; w_i <= R - 1; ++w_i) {
    for (int w_j = -R; w_j <= R; ++w_j) {
      hist0.AddElement(Get(input, w_i, w_j));
    }
  }

  for (int i = 0; i < input.rows; ++i) {
    for (int w_j = -R; w_j <= R; ++w_j) {
      hist0.RemoveElement(Get(input, i - R - 1, w_j));
      hist0.AddElement(Get(input, i + R, w_j));
    }

    Hist hist = hist0;

    for (int j = 0; j < input.cols; ++j) {
      output.at<uchar>(i, j) = (uchar)hist.Median();

      // update hist
      hist.SubtractHist(hists.at(j));
      hist.AddHist(hists.at(j + 2 * R + 1));
    }

    // update hists
    for (int j = -R; j < input.cols + R + 1; ++j) {
      hists.at(j + R).RemoveElement(Get(input, i - R, j));
      hists.at(j + R).AddElement(Get(input, i + R + 1, j));
    }
  }

  return output;
}

Mat MedianFilter(const Mat& input, int R, MedianFilterType type = MedianFilterType::kNaive) {
  std::vector<Mat> BGRin(3);
  split(input, BGRin);
  std::vector<Mat> BGRout(3);

  for (int i = 0; i < 3; ++i) {
    switch(type) {
      case MedianFilterType::kNaive: {
        BGRout[i] = MedianFilterNaive(BGRin[i], R);
        break;
      }
      case MedianFilterType::kHuang: {
        BGRout[i] = MedianFilterHuang(BGRin[i], R);
        break;
      }
      case MedianFilterType::kConstant: {
        BGRout[i] = MedianFilterConstant(BGRin[i], R);
        break;
      }
    }
  }

  Mat output;
  merge(BGRout, output);
  return output;
}

int main(int argc, char** argv) {
  using namespace cv;
  using namespace std::chrono;

  if (argc < 5) {
    std::cout << "usage: <bin> <input> <output> <R> <naive|huang|const|opencv>" << std::endl;
    return -1;
  }

  std::string input = argv[1];
  std::string output = argv[2];
  int R = strtol(argv[3], NULL, 10);
  std::string type_str = argv[4];

  Mat image = imread(input);
  if (image.empty())
  {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  MedianFilterType type;
  if (type_str == "naive") {
    type = MedianFilterType::kNaive;
  } else if (type_str == "huang") {
    type = MedianFilterType::kHuang;
  } else if (type_str == "const") {
    type = MedianFilterType::kConstant;
  }

  auto t0 = steady_clock::now();
  Mat output_image;
  if (type_str == "opencv") {
    medianBlur(image, output_image, 2 * R + 1);
  } else {
    output_image = MedianFilter(image, R, type);
  }
  auto t1 = steady_clock::now();

  std::cout << duration_cast<milliseconds>(t1 - t0).count() << std::endl;

  imwrite(output, output_image);
  return 0;
}
