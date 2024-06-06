#ifndef ATOMINTELMASH_PRACTICE_CALIBRATE_H
#define ATOMINTELMASH_PRACTICE_CALIBRATE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <armadillo>

//extern "C" {
//#include <lapacke.h>
//#include <cblas.h>
//}

struct Point {
    Point (int x, int y) : x(x), y(y) {}
    Point () {}
    int x;
    int y;

    friend std::ostream& operator<<(std::ostream& out, const Point& p){
        return out << "Point: " << p.x << " " << p.y;
    }
    static bool comp(Point p1, Point p2) {
        int epsilon = 20;
        return abs(p1.x - p2.x) > epsilon ? p1.x < p2.x : p1.y < p2.y;
    }
    static void sortPoint(std::vector<Point> & points) {
        std::sort(points.begin(), points.end(), Point::comp);
    }
};

struct intrinsicsCameraParam {
    double fx, fy, cx, cy;

    friend std::ostream& operator<<(std::ostream& out, const intrinsicsCameraParam& k){
        return out << k.fx << " " << "0" << " " << k.cx << std::endl
                << 0 << " " << k.fy << " " << k.cy << std::endl
                << 0 << " " << 0 << " " << 1;
    }

    std::vector<std::vector<double>> getMatK();
};

struct Distortion {
    double k1, k2;
};


struct Calibrate {
private:
    cv::Mat __GaussianBlur(cv::Mat const & img);
    std::vector<Point> __PrevittOperator(cv::Mat const & img);
    std::vector<Point> __delete_similar_points(std::vector<Point> const & points);
    arma::mat __findHomography(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints);
public:
    std::vector<Point> find_points(cv::Mat const & src);
    void draw_points(cv::Mat const & img, std::vector<Point> const & points);
    std::vector<Point> getObjectPoints(int deskSizeX, int deskSizeY, int sizeSquare);
    intrinsicsCameraParam getK(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints);

    double findRadialDistortion(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img);
    double findRadialDistortion2(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img);
    Distortion findRadialDistortionDouble(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img);

    cv::Mat unDistort(cv::Mat const & img, double k);
};

#endif //ATOMINTELMASH_PRACTICE_CALIBRATE_H
