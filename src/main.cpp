#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include "calibrate.h"

#include <opencv2/opencv.hpp>

std::string PATH = "../img/";

// 6 4 40
// 10 7 25
int DESK_SIZE_X = 10;
int DESK_SIZE_Y = 7;
int SIZE_SQUARE = 25;


int main(int argc, char const* argv[]) {

    std::vector<cv::Mat> images;
    for (const auto & file : fs::directory_iterator(PATH)) {
        cv::Mat img = cv::imread(file.path(), cv::IMREAD_GRAYSCALE);
        images.push_back(img);
    }

    Calibrate calibrate;

    cv::Mat image = images[4];
    std::vector<Point> imagePoints = calibrate.find_points(image);
    Point::sortPoint(imagePoints);
    calibrate.draw_points(image, imagePoints);

    std::vector<Point> objectPoints = calibrate.getObjectPoints(DESK_SIZE_X, DESK_SIZE_Y, SIZE_SQUARE);
//    int count_obj_points = 0;
//    for (int i = 0; i < objectPoints.size(); i += 1) {
//        std::cout << "Object " << objectPoints[i] << std::endl;
//        ++count_obj_points;
//    }
//    std::cout << "Count Object point: " << count_obj_points << std::endl;
//
//    int count_img_points = 0;
//    for (int i = 0; i < imagePoints.size(); i += 1) {
//        std::cout << "Image " << imagePoints[i] << std::endl;
//        ++count_img_points;
//    }
//    std::cout << "Count Image point: " << count_img_points << std::endl;

//    Point p1(10, 20);
//    Point p2(50, 60);
//    std::vector<Point> u = {p1};
//    std::vector<Point> v = {p2};

//    K intrinsics_camera_param = calibrate.getK(u, v);
    intrinsicsCameraParam intrinsics_camera_param = calibrate.getK(imagePoints, objectPoints);

//    std::vector<std::vector<double>> k_inv;
//    cv::invert(intrinsics_camera_param.getMatK(), k_inv);
//
//
//    for (int i = 0; i < k_inv.size(); ++i) {
//        for (int j = 0; j < k_inv[0].size(); ++j) {
//            std::cout << k_inv[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

//    cv::Mat k_image;
//    cv::warpPerspective(image, k_image, )

//    double k = calibrate.findRadialDistortion(intrinsics_camera_param, imagePoints, objectPoints, image);
//    std::cout << "RadialDistortion: " << k << std::endl;

//    double k = calibrate.findRadialDistortion2(intrinsics_camera_param, imagePoints, objectPoints, image);
//    std::cout << "RadialDistortion: " << k << std::endl;

//    Distortion dist = calibrate.findRadialDistortionDouble(intrinsics_camera_param, imagePoints, objectPoints, image);
//    std::cout << "RadialDistortion k1: " << dist.k1 << std::endl;
//    std::cout << "RadialDistortion k2: " << dist.k2 << std::endl;

//    Distortion dist = calibrate.findRadialDistortionMax(intrinsics_camera_param, imagePoints, objectPoints, image);
//    std::cout << "RadialDistortion k1: " << dist.k1 << std::endl;
//    std::cout << "RadialDistortion k2: " << dist.k2 << std::endl;
//    std::cout << "RadialDistortion k3: " << dist.k3 << std::endl;
//    std::cout << "RadialDistortion k4: " << dist.k4 << std::endl;

    double k = calibrate.findNewDistortion(intrinsics_camera_param, imagePoints, objectPoints, image);

    cv::Mat undistortedMat = calibrate.unDistort(image, k);
//    cv::Mat undistortedMat = calibrate.doubleUnDistort(image, dist);
//    cv::Mat undistortedMat = calibrate.maxUnDistort(image, dist);

//    cv::Point p1(83, 0), p2(83, image.rows - 1);
//    cv::line(undistortedMat, p1, p2, cv::Scalar(255, 0, 0), 1);
//    cv::Point p3(0, 73), p4(image.cols-1, 73);
//    cv::line(undistortedMat, p3, p4, cv::Scalar(255, 0, 0), 1);

    cv::imshow("undistortedMat", undistortedMat);
    cv::waitKey();

    return 0;
}


