#include "calibrate.h"

cv::Mat Calibrate::__GaussianBlur(cv::Mat const & img) {
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            float k1 = 0.0625;
            float k2 = 0.125;
            float k3 = 0.0625;
            float k4 = 0.125;
            float k5 = 0.25;
            float k6 = 0.125;
            float k7 = 0.0625;
            float k8 = 0.125;
            float k9 = 0.0625;

            int p1 = img.at<uchar>(y - 1, x - 1);
            int p2 = img.at<uchar>(y - 1, x);
            int p3 = img.at<uchar>(y - 1, x + 1);
            int p4 = img.at<uchar>(y, x - 1);
            int p5 = img.at<uchar>(y, x);
            int p6 = img.at<uchar>(y, x + 1);
            int p7 = img.at<uchar>(y + 1, x - 1);
            int p8 = img.at<uchar>(y + 1, x);
            int p9 = img.at<uchar>(y + 1, x + 1);

            res.at<uchar>(y, x) = k1*p1 + k2*p2 + k3*p3 + k4*p4 + k5*p5 + k6*p6 + k7*p7 + k8*p8 + k9*p9;
        }
    }
    return res;
}

std::vector<Point> Calibrate::__PrevittOperator(cv::Mat const & img) {
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            res.at<uchar>(y, x) = img.at<uchar>(y, x);
        }
    }

    std::vector<std::vector<int>> Gx(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Gy(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Hp(img.cols, std::vector<int>(img.rows, 0));
    std::vector<Point> points;

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            int z1 = img.at<uchar>(y - 1, x - 1);
            int z2 = img.at<uchar>(y - 1, x);
            int z3 = img.at<uchar>(y - 1, x + 1);
            int z4 = img.at<uchar>(y, x - 1);
            int z6 = img.at<uchar>(y, x + 1);
            int z7 = img.at<uchar>(y + 1, x - 1);
            int z8 = img.at<uchar>(y + 1, x);
            int z9 = img.at<uchar>(y + 1, x + 1);

            int gx = (z7 + z8 + z9) - (z1 + z2 + z3);
            int gy = (z3 + z6 + z9) - (z1 + z4 + z7);
            Gx[x][y] = gx;
            Gy[x][y] = gy;
        }
    }

    for (int x = 10; x < Gx.size() - 10; ++x) {
        for (int y = 10; y < Gx[0].size() - 10; ++y) {
            float k = 0.2; // 0.2
            int gp1 = 0;
            int gp2 = 0;
            int gp3 = 0;

            for (int i = x - 1; i < x + 2; ++i) {
                for (int j = y - 1; j < y + 2; ++j) {
                    int gx = Gx[i][j];
                    int gy = Gy[i][j];
                    gp1 += gx*gx;
                    gp2 += gx*gy;
                    gp3 += gy*gy;
                }
            }

            int hp = (gp1 * gp3 - gp2*gp2) - k*(gp1 + gp3)*(gp1 + gp3);
            Hp[x][y] = hp;

            if (hp > 1000000) {
                Point p = Point(x, y);
                points.push_back(p);
            }
        }
    }

    return points;
}

std::vector<Point> Calibrate::__delete_similar_points(std::vector<Point> const & points) {
    uint accuracy = 10;
    std::vector<Point> res;

    for (int i = 0; i < points.size(); ++i) {
        bool is_similar = false;
        for (int j = 0; j < i; ++j) {
            if (std::abs(points[i].x - points[j].x) < accuracy && std::abs(points[i].y - points[j].y) < accuracy) {
                is_similar = true;
            }
        }
        if ( ! is_similar ) {
            res.push_back(points[i]);
        }
    }
    return res;
}

std::vector<Point> Calibrate::find_points(cv::Mat const & src) {
    cv::Mat img;

    img = __GaussianBlur(src);
    for (int i = 0; i < 10; ++i) { // 10
        img = __GaussianBlur(img);
    }

    std::vector<Point> points = __PrevittOperator(img);

    points = __delete_similar_points(points);

    Point::sortPoint(points);

    return points;
}

void Calibrate::draw_points(cv::Mat const & img, std::vector<Point> const & points) {
    uint count_points = 0;
    for (int i = 0; i < points.size(); ++i) {
        std::cout << points[i] << std::endl;
        ++count_points;
    }
    std::cout << "Count points: " << count_points << std::endl;

    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            res.at<uchar>(y, x) = img.at<uchar>(y, x);
        }
    }

    for (int i = 0; i < points.size(); ++ i) {
        Point p = points[i];
        cv::Point centerCircle(p.x, p.y);
        cv::Scalar colorCircle(0);
        cv::circle(res, centerCircle, 10, colorCircle, cv::FILLED);
    }

    cv::imshow("draw_points", res);
    cv::waitKey();
}


std::vector<Point> Calibrate::getObjectPoints(int deskSizeX, int deskSizeY, int sizeSquare) {
    std::vector<Point> res;
    for (int x = 0; x < deskSizeX; x += 1) {
        for (int y = 0; y < deskSizeY; y += 1) {
            res.push_back(Point(x * sizeSquare, y * sizeSquare));
        }
    }
    return res;
}

arma::mat Calibrate::__findHomography(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints) {
    if (objectPoints.size() != imagePoints.size()) {
        arma::mat H;
        return H;
    }

    std::vector<int> P, u;

    arma::mat Q(objectPoints.size() * 2, 9);

    for (int i = 0; i < objectPoints.size(); ++i) {
        Q.col(0).row(2 * i) = - objectPoints[i].x;
        Q.col(1).row(2 * i) = 0;
        Q.col(2).row(2 * i) = objectPoints[i].x * imagePoints[i].x;
        Q.col(3).row(2 * i) = -objectPoints[i].y;
        Q.col(4).row(2 * i) = 0;
        Q.col(5).row(2 * i) = objectPoints[i].y * imagePoints[i].x;
        Q.col(6).row(2 * i) = -1;
        Q.col(7).row(2 * i) = 0;
        Q.col(8).row(2 * i) = imagePoints[i].x;
        Q.col(0).row(2 * i + 1) = 0;
        Q.col(1).row(2 * i + 1) = -objectPoints[i].x;
        Q.col(2).row(2 * i + 1) = objectPoints[i].x * imagePoints[i].y;
        Q.col(3).row(2 * i + 1) = 0;
        Q.col(4).row(2 * i + 1) = - objectPoints[i].y;
        Q.col(5).row(2 * i + 1) = objectPoints[i].y * imagePoints[i].y;
        Q.col(6).row(2 * i + 1) = 0;
        Q.col(7).row(2 * i + 1) = -1;
        Q.col(8).row(2 * i + 1) = imagePoints[i].y;
    }

    arma::mat U;
    arma::vec S;
    arma::mat V;

//    std::cout << Q << std::endl;

    arma::svd(U, S, V, Q);

//    V = V.t();
    arma::mat H = V.col(8);

    H = reshape(H, 3, 3);
    return H;
}


arma::mat newFindHomography(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints) {
    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;
    double rz = 300;

    int countPoints = objectPoints.size();

    arma::mat RT(3, 4);
    arma::mat v(4, countPoints);
    arma::mat u(3, countPoints);

    for (int i = 0; i < countPoints; ++i) {
        u.at(0, i) = imagePoints[i].x;
        u.at(1, i) = imagePoints[i].y;
        u.at(2, i) = 1;
    }
    for (int i = 0; i < countPoints; ++i) {
        v.at(0, i) = objectPoints[i].x;
        v.at(1, i) = objectPoints[i].y;
        v.at(2, i) = 0;
        v.at(3, i) = 1;
    }
    RT.at(0, 0) = 1;
    RT.at(0, 1) = 0;
    RT.at(0, 2) = 0;
    RT.at(0, 3) = rx;
    RT.at(1, 0) = 0;
    RT.at(1, 1) = 1;
    RT.at(1, 2) = 0;
    RT.at(1, 3) = ry;
    RT.at(2, 0) = 0;
    RT.at(2, 1) = 0;
    RT.at(2, 2) = 1;
    RT.at(2, 3) = rz;

    arma::mat H(3, 3);
    H = arma::solve(u, RT * v);

    H.print("find^^^: ");
}


intrinsicsCameraParam Calibrate::getK(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints) {

    arma::mat H = this->__findHomography(imagePoints, objectPoints);

//    arma::mat H = newFindHomography(imagePoints, objectPoints);

    H = H / H.at(2, 2);

    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;

    double rz = 175;

    intrinsicsCameraParam res;
    res.fx = H.at(0, 0) ;
    res.fy = H.at(1, 1) ;
    res.cx = abs(H.at(0, 2) - rx);
    res.cy = abs(H.at(1, 2) - ry);
//    res.cx = abs(H.at(0, 2));
//    res.cy = abs(H.at(1, 2));

    std::cout << H << std::endl;
    std::cout << res << std::endl;

    return res;
}


std::vector<std::vector<double>> intrinsicsCameraParam::getMatK() {
    std::vector<double> matK_1(3, 0);
    matK_1[0] = fx;
    matK_1[2] = cx;
    std::vector<double> matK_2(3, 0);
    matK_2[1] = fy;
    matK_2[2] = cy;
    std::vector<double> matK_3(3, 0);
    matK_3[2] = 1;

    std::vector<std::vector<double>> matK;
    matK.push_back(matK_1);
    matK.push_back(matK_2);
    matK.push_back(matK_3);

    for (int i = 0; i < matK.size(); ++i) {
        for (int j = 0; j < matK[0].size(); ++j) {
            std::cout << matK[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return matK;
}


double Calibrate::findRadialDistortion(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img) {
    cv::Mat res;
    cv::cvtColor(img, res, cv::COLOR_GRAY2BGR);
//    double from = -0.0000067;

//    std::vector<Point> cloneImagePoints = objectPoints;
//    int new_x = (x - x0) / (1 + k * r*r) + x0;
//    int new_y = (y - y0) / (1 + k * r*r) + y0;

// k = -2e-7;
    double from = 1e-3;
    double to = 0;
    double step = 1e-7;

//    double from = 0;
//    double to = 1;
//    double step = 10;

    double min = 0;
    double min_k = 0;
    double x0 = img.cols / 2.;
    double y0 = img.rows / 2.;
    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;
    double k = from;
    while (k < to) {
        double minimize = 0;
        for (int i = 0; i < objectPoints.size(); ++i) {
            double x_img = imagePoints[i].x;
            double y_img = imagePoints[i].y;

            double mx = imagePoints[i].x * cameraParam.fx + rx + cameraParam.cx;
            double my = imagePoints[i].y * cameraParam.fy + ry + cameraParam.cy;
//            double mx = cloneImagePoints[i].x;
//            double my = cloneImagePoints[i].y;

            double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

//            double x = (mx) * (1 + k * r*r);
//            double y = (my) * (1 + k * r*r);

            double x = (mx-x0) * (1 + k * r*r) + x0;
            double y = (my-y0) * (1 + k * r*r) + y0;

//            Point point = objectPoints[i];
//            double temp_x = point.x + imagePoints[0].x;
//            double temp_y = point.y + imagePoints[0].y;
//            double r = (temp_x - x0) * (temp_x - x0) + (temp_y - y0) * (temp_y - y0);
//            double xd = temp_x * (1 + k * r);
//            double yd = temp_y * (1 + k * r);
//
//
//            double x = (cameraParam.fx * xd + cameraParam.cx);
//            double y = (cameraParam.fy * yd + cameraParam.cy);

//            double x = (cameraParam.fx * (point.x - imagePoints[0].x) + cameraParam.cx);
//            double y = (cameraParam.fy * (point.y - imagePoints[0].y) + cameraParam.cy);
//            double x = point.x;
//            double y = point.y;

//            double r = (x - x0) * (x - x0) + (y - y0) * (y - y0);
//            double xd = x * (1 + k * r);
//            double yd = y * (1 + k * r);

//            minimize += ((x_img - xd) * (x_img - xd) + (y_img - yd) * (y_img - yd));
            minimize += ((x_img - x) * (x_img - x) + (y_img - y) * (y_img - y));

//            cv::Point centerCircle2(x_img, y_img);
//            cv::Scalar colorCircle2(0, 255, 0);
//            cv::circle(res, centerCircle2, 5, colorCircle2, cv::FILLED);
//
//            cv::Point centerCircle1(x, y);
//            cv::Scalar colorCircle1(0, 0, 255);
//            cv::circle(res, centerCircle1, 10, colorCircle1, cv::FILLED);
//
//            cv::imshow("draw_points", res);
//            cv::waitKey();
        }
        if (minimize < min || k == from) {
            min = minimize;
            min_k = k;

//            for (int i = 0; i < imagePoints.size(); ++i) {
//                double x = cloneImagePoints[i].x;
//                double y = cloneImagePoints[i].y;
//                long r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
//
//                int new_x = (x - x0) * (1 + min_k * r*r) + x0;
//                int new_y = (y - y0) * (1 + min_k * r*r) + y0;
//
//                cloneImagePoints[i].x = new_x;
//                cloneImagePoints[i].y = new_y;
//            }
        }
        k += step;
    }

    std::cout << "min " << ": " << min << std::endl;

    for (int i = 0; i < objectPoints.size(); ++i) {
        double x_img = imagePoints[i].x;
        double y_img = imagePoints[i].y;

        double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
        double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;
//        double mx = objectPoints[i].x * cameraParam.fx + rx;
//        double my = objectPoints[i].y * cameraParam.fy + ry;

        double mx_ = objectPoints[i].x * cameraParam.fx + rx;
        double my_ = objectPoints[i].y * cameraParam.fy + ry;

        double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

//        std::cout << "mx: " << mx  << " x0: " << x0 << " r: " << r  << " min_k: " << min_k << std::endl;

//        double x = (mx) * (1 + k * r*r);
//        double y = (my) * (1 + k * r*r);

        double x = (mx-x0) / (1 + min_k * r*r) + x0;
        double y = (my-y0) / (1 + min_k * r*r) + y0;

//        std::cout << "x: " << x << std::endl;
//        std::cout << "y: " << y << std::endl;

        cv::Point centerCircle1(x, y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(res, centerCircle1, 5, colorCircle1, cv::FILLED);

        cv::Point centerCircle2(x_img, y_img);
        cv::Scalar colorCircle2(0, 255, 0);
        cv::circle(res, centerCircle2, 5, colorCircle2, cv::FILLED);

        cv::Point centerCircle3(x0, y0);
        cv::Scalar colorCircle3(255, 0, 0);
        cv::circle(res, centerCircle3, 10, colorCircle3, cv::FILLED);

        cv::Point centerCircle4(mx, my);
        cv::Scalar colorCircle4(0, 255, 255);
        cv::circle(res, centerCircle4, 2, colorCircle4, cv::FILLED);

//        cv::Point centerCircle1(mx_, my_);
//        cv::Scalar colorCircle1(0, 0, 255);
//        cv::circle(res, centerCircle1, 5, colorCircle1, cv::FILLED);


//        cv::imshow("draw_points", res);
//        cv::waitKey();
    }

//    cv::waitKey();
    return min_k;
}


double Calibrate::findRadialDistortion2(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img) {
    cv::Mat res;
    cv::cvtColor(img, res, cv::COLOR_GRAY2BGR);

    double from = -1e-12;
    double to = 0;
    double step = 1e8  ;
    double min = 0;
    double min_k = 0;
    double x0 = img.cols / 2.;
    double y0 = img.rows / 2.;
    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;
    double k = from;
    while (k < to) {
        double minimize = 0;
        for (int i = 0; i < objectPoints.size(); ++i) {
            double x_img = imagePoints[i].x;
            double y_img = imagePoints[i].y;

            double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
            double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;

            double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

            double x = (mx-x0) * (1 + k * r*r*r*r) + x0;
            double y = (my-y0) * (1 + k * r*r*r*r) + y0;

            minimize += ((x_img - x) * (x_img - x) + (y_img - y) * (y_img - y));
        }
        if (minimize < min || k == from) {
            min = minimize;
            min_k = k;
        }
        k += step;
    }

    for (int i = 0; i < objectPoints.size(); ++i) {
        double x_img = imagePoints[i].x;
        double y_img = imagePoints[i].y;

        double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
        double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;
        double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

//        std::cout << "mx: " << mx  << " x0: " << x0 << " r: " << r  << " min_k: " << min_k << std::endl;

        double x = (mx-x0) * (1 + min_k * r*r*r*r) + x0;
        double y = (my-y0) * (1 + min_k * r*r*r*r) + y0;

//        std::cout << "x: " << x << std::endl;
//        std::cout << "y: " << y << std::endl;

        cv::Point centerCircle1(x, y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(res, centerCircle1, 5, colorCircle1, cv::FILLED);

        cv::Point centerCircle2(x_img, y_img);
        cv::Scalar colorCircle2(0, 255, 0);
        cv::circle(res, centerCircle2, 5, colorCircle2, cv::FILLED);

        cv::Point centerCircle3(x0, y0);
        cv::Scalar colorCircle3(255, 0, 0);
        cv::circle(res, centerCircle3, 10, colorCircle3, cv::FILLED);

        cv::Point centerCircle4(mx, my);
        cv::Scalar colorCircle4(0, 255, 255);
        cv::circle(res, centerCircle4, 2, colorCircle4, cv::FILLED);

//        cv::imshow("draw_points", res);
//        cv::waitKey();
    }
    cv::waitKey();

    return min_k;
}


Distortion Calibrate::findRadialDistortionDouble(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img) {
    Distortion distortion;
    cv::Mat res;
    cv::cvtColor(img, res, cv::COLOR_GRAY2BGR);
    double min = 0;
    double min_k1 = 0;
    double min_k2 = 0;
    double x0 = img.cols / 2.;
    double y0 = img.rows / 2.;
    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;

    double from1 = -1e-6;
    double to1 = 0;
    double step1 = 1e-8;
    double from2 = -1e-12;
    double to2 = 1e-12;
    double step2 = 1e-13;

    double k1 = from1;
    while (k1 < to1) {
        double k2 = from2;
        while (k2 < to2) {
            double minimize = 0;
            for (int i = 0; i < objectPoints.size(); ++i) {
                double x_img = imagePoints[i].x;
                double y_img = imagePoints[i].y;

                double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
                double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;

                double r = sqrt((mx - x0) * (mx - x0) + (my - y0) * (my - y0));

                double x = (mx - x0) * (1 + k1*r*r + k2*r*r*r*r) + x0;
                double y = (my - y0) * (1 + k1*r*r + k2*r*r*r*r) + y0;

                minimize += ((x_img - x) * (x_img - x) + (y_img - y) * (y_img - y));
            }

            if (minimize < min || k1 == from1) {
                min = minimize;
                min_k1 = k1;
                min_k2 = k2;
            }

            k2 += step2;
        }

        k1 += step1;
    }

    std::cout << "min: " << min << std::endl;


    for (int i = 0; i < objectPoints.size(); ++i) {
        double x_img = imagePoints[i].x;
        double y_img = imagePoints[i].y;

        double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
        double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;

        double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

        double x = (mx-x0) * (1 + min_k1 * r*r + min_k2 * r*r*r*r) + x0;
        double y = (my-y0) * (1 + min_k1 * r*r + min_k2 * r*r*r*r) + y0;

        cv::Point centerCircle1(x, y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(res, centerCircle1, 5, colorCircle1, cv::FILLED);

        cv::Point centerCircle2(x_img, y_img);
        cv::Scalar colorCircle2(0, 255, 0);
        cv::circle(res, centerCircle2, 5, colorCircle2, cv::FILLED);

        cv::Point centerCircle3(x0, y0);
        cv::Scalar colorCircle3(255, 0, 0);
        cv::circle(res, centerCircle3, 10, colorCircle3, cv::FILLED);

        cv::Point centerCircle4(mx, my);
        cv::Scalar colorCircle4(0, 255, 255);
        cv::circle(res, centerCircle4, 2, colorCircle4, cv::FILLED);

        cv::imshow("draw_points", res);
//        cv::waitKey();
    }
    cv::waitKey();

    distortion.k1 = min_k1;
    distortion.k2 = min_k2;

    return distortion;
}

Distortion Calibrate::findRadialDistortionMax(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img) {
    Distortion distortion;
    cv::Mat res;
    cv::cvtColor(img, res, cv::COLOR_GRAY2BGR);
    double min = 0;
    double min_k1 = 0;
    double min_k2 = 0;
    double min_k3 = 0;
    double min_k4 = 0;
    double x0 = img.cols / 2.;
    double y0 = img.rows / 2.;
    double rx = imagePoints[0].x;
    double ry = imagePoints[0].y;

    double from1 = -1e-6;
    double to1 = 1e-6;
    double step1 = 1e-7;

    double from2 = -1e-9;
    double to2 = 1e-9;
    double step2 = 1e-10;

    double from3 = -1e-6;
    double to3 = 1e-6;
    double step3 = 1e-7;

    double from4 = -1e-9;
    double to4 = 1e-9;
    double step4 = 1e-10;

    double k1 = from1;
    while (k1 < to1) {
        double k2 = from2;
        while (k2 < to2) {
            double k3 = from3;
            while (k3 < to3) {
                double k4 = from4;
                while (k4 < to4) {
                    double minimize = 0;
                    for (int i = 0; i < objectPoints.size(); ++i) {
                        double x_img = imagePoints[i].x;
                        double y_img = imagePoints[i].y;

                        double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
                        double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;

                        double r = sqrt((mx - x0) * (mx - x0) + (my - y0) * (my - y0));

                        double x = (mx - x0) * (1 + k1*r*r + k2*r*r*r*r) + x0;
                        double y = (my - y0) * (1 + k3*r*r + k4*r*r*r*r) + y0;

                        minimize += ((x_img - x) * (x_img - x) + (y_img - y) * (y_img - y));
                    }

                    if (minimize < min || k1 == from1) {
                        min = minimize;
                        min_k1 = k1;
                        min_k2 = k2;
                        min_k3 = k3;
                        min_k4 = k4;
                    }

                    k4 += step4;
                }

                k3 += step3;
            }

            k2 += step2;
        }

        k1 += step1;
    }

    std::cout << "min: " << min << std::endl;


    for (int i = 0; i < objectPoints.size(); ++i) {
        double x_img = imagePoints[i].x;
        double y_img = imagePoints[i].y;

        double mx = objectPoints[i].x * cameraParam.fx + rx + cameraParam.cx;
        double my = objectPoints[i].y * cameraParam.fy + ry + cameraParam.cy;

        double r = sqrt((mx - x0)*(mx - x0) + (my - y0)*(my - y0));

        double x = (mx-x0) * (1 + min_k1 * r*r + min_k2 * r*r*r*r) + x0;
        double y = (my-y0) * (1 + min_k3 * r*r + min_k4 * r*r*r*r) + y0;

        cv::Point centerCircle1(x, y);
        cv::Scalar colorCircle1(0, 0, 255);
        cv::circle(res, centerCircle1, 5, colorCircle1, cv::FILLED);

        cv::Point centerCircle2(x_img, y_img);
        cv::Scalar colorCircle2(0, 255, 0);
        cv::circle(res, centerCircle2, 5, colorCircle2, cv::FILLED);

        cv::Point centerCircle3(x0, y0);
        cv::Scalar colorCircle3(255, 0, 0);
        cv::circle(res, centerCircle3, 10, colorCircle3, cv::FILLED);

        cv::Point centerCircle4(mx, my);
        cv::Scalar colorCircle4(0, 255, 255);
        cv::circle(res, centerCircle4, 2, colorCircle4, cv::FILLED);

//        cv::imshow("draw_points", res);
//        cv::waitKey();
    }
    cv::waitKey();

    distortion.k1 = min_k1;
    distortion.k2 = min_k2;
    distortion.k3 = min_k3;
    distortion.k4 = min_k4;

    return distortion;
}



cv::Mat Calibrate::unDistort(cv::Mat const & img, double k) {
//    cv::Mat res = img.clone();
////    k = -k;
//
//    if (k == 0) {
//        return res;
//    }
//
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            res.at<uchar>(y, x) = 0;
//        }
//    }
//
//    int x0 = img.cols / 2;
//    int y0 = img.rows / 2;
//
//    for (int y = 0; y < img.rows; ++y) {
//        for (int x = 0; x < img.cols; ++x) {
//            uchar value = img.at<uchar>(y, x);
//
//            long r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
//
//            double rd = (1/(k*r) + sqrt(1/(k*k*r*r) - 4/k))/2;
//
////            int new_x = (x - x0) / (1 + k * rd*rd) + x0;
////            int new_y = (y - y0) / (1 + k * rd*rd) + y0;
//            int new_x = (x - x0) * (r/rd) + x0;
//            int new_y = (y - y0) * (r/rd) + y0;
//
////            std::cout << "new_x: " << new_x << "; new_y: " << new_y << std::endl;
//
//            if ((new_x > 0 && new_x < img.cols) && (new_y > 0 && new_y < img.rows)) {
//                res.at<uchar>(new_y, new_x) = value;
//            }
//        }
//    }
//
//    return res;



    cv::Mat res = img.clone();

    if (k == 0) {
        return res;
    }

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            res.at<uchar>(y, x) = 0;
        }
    }

    int x0 = img.cols / 2;
    int y0 = img.rows / 2;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            uchar value = img.at<uchar>(y, x);

            long r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

            double rd = (1/(k*r) + sqrt(1/(k*k*r*r) - 4/k))/2;

//            int new_x = (x - x0) * (1 + k * r*r) + x0;
//            int new_y = (y - y0) * (1 + k * r*r) + y0;
            int new_x = (x - x0) * (rd / r) + x0;
            int new_y = (y - y0) * (rd / r) + y0;

//            std::cout << "new_x: " << new_x << "; new_y: " << new_y << std::endl;

            if ((new_x > 0 && new_x < img.cols) && (new_y > 0 && new_y < img.rows)) {
                res.at<uchar>(y, x) = img.at<uchar>(new_y, new_x);;
            }
        }
    }

    return res;
}


cv::Mat Calibrate::doubleUnDistort(cv::Mat const & img, Distortion distortion) {
    cv::Mat res = img.clone();

    if (distortion.k1 == 0 || distortion.k2 == 0) {
        return res;
    }

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            res.at<uchar>(y, x) = 0;
        }
    }

    int x0 = img.cols / 2;
    int y0 = img.rows / 2;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            uchar value = img.at<uchar>(y, x);

            long r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

//            double rd = (1/(k*r) + sqrt(1/(k*k*r*r) - 4/k))/2;

            int new_x = (x - x0) / (1 + distortion.k1 * r*r + distortion.k2 * r*r*r*r) + x0;
            int new_y = (y - y0) / (1 + distortion.k1 * r*r + distortion.k2 * r*r*r*r) + y0;
//            int new_x = (x - x0) * (rd / r) + x0;
//            int new_y = (y - y0) * (rd / r) + y0;

//            std::cout << "new_x: " << new_x << "; new_y: " << new_y << std::endl;

            if ((new_x > 0 && new_x < img.cols) && (new_y > 0 && new_y < img.rows)) {
                res.at<uchar>(y, x) = img.at<uchar>(new_y, new_x);;
            }
        }
    }

    return res;
}


cv::Mat Calibrate::maxUnDistort(cv::Mat const & img, Distortion distortion) {
    cv::Mat res = img.clone();

    if (distortion.k1 == 0 || distortion.k2 == 0) {
        return res;
    }

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            res.at<uchar>(y, x) = 0;
        }
    }

    int x0 = img.cols / 2;
    int y0 = img.rows / 2;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            long r = sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

            int new_x = (x - x0) / (1 + distortion.k1 * r*r + distortion.k2 * r*r*r*r) + x0;
            int new_y = (y - y0) / (1 + distortion.k3 * r*r + distortion.k4 * r*r*r*r) + y0;

            if ((new_x > 0 && new_x < img.cols) && (new_y > 0 && new_y < img.rows)) {
                res.at<uchar>(y, x) = img.at<uchar>(new_y, new_x);;
            }
        }
    }

    return res;
}


double Calibrate::findNewDistortion(intrinsicsCameraParam const & cameraParam, std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints, cv::Mat img) {
    int x0 = img.cols / 2;
    int y0 = img.rows / 2;
    int countPoints = imagePoints.size();

    arma::mat M(countPoints, 3);

    arma::mat b(countPoints, 1);

    double lambda;

    for (int i = 0; i < countPoints; ++i) {
        int x = imagePoints[i].x;
        int y = imagePoints[i].y;

        M.at(i, 0) = x;
        M.at(i, 1) = y;
        M.at(i, 2) = 1;

        b.at(i, 0) = - (x*x + y*y);
    }

    arma::mat abc(3, 3);
//    abc.print("abc: ");
    abc = arma::solve(M.t() * M, arma::eye(3, 3)) * M.t() * b;

    std::cout << "hi " << std::endl << abc.at(0, 0) << std::endl << abc.at(0, 1) << std::endl << abc.at(0, 2) << std::endl;

    lambda = 1 / (x0*x0 + y0*y0 + abc.at(0, 0)*x0 + abc.at(0, 1)*y0 + abc.at(0, 2));

    std::cout << "hello lambda: " << lambda << std::endl;

    return lambda;
}


