#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

// Solve system of linear equations using least squares normal equations
Eigen::MatrixXf solveLeastSquares(const Eigen::MatrixXf& pts2d,
                                         const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 11;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b(rows, 1);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z;
        b.row(i) << x;
        b.row(i + 1) << y;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Append a 1 to the end since scale is constant
    sol.conservativeResize(sol.rows() + 1, sol.cols());
    sol(sol.rows() - 1, 0) = 1;
    return sol;
}

// Overload of the above function using cv::Mat instead of Eigen

cv::Mat solveLeastSquares(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Wrap OpenCV matrices in Eigen::Map
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveLeastSquares(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}

// Solve system of linear equations using singular value decomposition
Eigen::MatrixXf solveSVD(const Eigen::MatrixXf& pts2d, const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 12;
    Eigen::MatrixXf A(rows, cols);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z, -x;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y;
    }

    // Compute the orthogonal matrix of eigenvectors of A_T*A
    Eigen::MatrixXf eigenvectors = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    // Get the eigenvector with the smallest eigenvalue (the last column of the V matrix)
    Eigen::MatrixXf smallest = eigenvectors.col(eigenvectors.cols() - 1);
    // std::cout << "sol = \n" << smallest << std::endl;
    return smallest;
}

// Overload of the above function using cv::Mat instead of Eigen
cv::Mat solveSVD(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Convert OpenCV matrices to Eigen matrices
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveSVD(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}
// Project a set of points from 3D space to the 2D image plane
cv::Mat project3D(const cv::Mat& projMat, const cv::Mat& pt3d) {
    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * pt3d;
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }

    return projected;
}

void runProblem1a() {
    cv::Mat _pts3DNorm = (cv::Mat_<float>(6,3) <<
            1.5706,-0.1490,0.2598,
            -1.5282, 0.9695, 0.3802,
            -0.6821, 1.2856, 0.4078,
            0.4124, -1.0201, -0.0915,
            1.2095, 0.2812, -0.1280,
            0.8819, -0.8481, 0.5255);

    cv::Mat _picANorm = (cv::Mat_<float>(6,2) <<
                                              1.0486, -0.3645,
                                              -1.6851, -0.4004,
                                              -0.9437, -0.4200,
                                              1.0682, 0.0699,
                                              0.6077, -0.0771,
                                              1.2543, -0.6454);

    cv::Mat _picA, _picB, _pts3D;

    // Get the last 3D normalized point so we can check our m matrix later
    cv::Mat lastPt3D = _pts3DNorm.col(_pts3DNorm.cols - 1);
    lastPt3D.push_back(1.f);
    cv::Mat lastPt2D = _picANorm.col(_picANorm.cols - 1);

    // Row-vector version of the last point so that we can print it more easily
    cv::Mat lastPt3D_T;
    cv::transpose(lastPt3D, lastPt3D_T);

    // Compute projection matrix
    {
        auto sol = solveLeastSquares(_picANorm, _pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
//        logger->info("Calibration parameters (using normal least squares):\n{}\nProjected "
//                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
//                     FormattedMat(params),
//                     lastPt3D_T,
//                     projection_T,
//                     residual);
    }

    {
        auto sol = solveSVD(_picANorm, _pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
//        logger->info("Calibration parameters (using singu    Config config(CONFIG_FILE_PATH);
//lar value decomposition):\n{}\nProjected "
//                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
//                     FormattedMat(params),
//                     lastPt3D_T,
//                     projection_T,
//                     residual);
    }
}

using namespace std;

int main() {
    Eigen::MatrixXf pts2d(6,2), pts3d(6,3);
    pts3d <<
        1.5706,-0.1490,0.2598,
        -1.5282, 0.9695, 0.3802,
        -0.6821, 1.2856, 0.4078,
        0.4124, -1.0201, -0.0915,
        1.2095, 0.2812, -0.1280,
        0.8819, -0.8481, 0.5255;

    pts2d <<
        1.0486, -0.3645,
        -1.6851, -0.4004,
        -0.9437, -0.4200,
        1.0682, 0.0699,
        0.6077, -0.0771,
        1.2543, -0.6454;


    std::cout <<pts3d.rows()  << "," << pts3d.cols() << std::endl;
    std::cout <<pts2d.rows()  << "," << pts2d.cols() << std::endl;

    std::cout << pts3d << endl;
    std::cout << pts2d << endl;

    assert(pts2d.rows() == pts3d.rows() && pts2d.cols() == 2 && pts3d.cols() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.rows() * 2;
    const size_t cols = 11;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b(rows, 1);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(i / 2, 0);
        float Y = pts3d(i / 2, 1);
        float Z = pts3d(i / 2, 2);
        float x = pts2d(i / 2, 0);
        float y = pts2d(i / 2, 1);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z;
        b.row(i) << x;
        b.row(i + 1) << y;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Append a 1 to the end since scale is constant
    sol.conservativeResize(sol.rows() + 1, sol.cols());
    sol(sol.rows() - 1, 0) = 1;

    cv::Mat cvSol;
    cv::eigen2cv(sol, cvSol);
    cv::Mat projMat = cvSol.reshape(0, 3);

    cv::Mat lastPt2D = pts2d.row(0);
    cv::Mat lastPt3D = pts3d.row(0);
    lastPt3D.push_back(1.f);


    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * lastPt3D;
    
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }
    
    // Compute residual
    double residual = cv::norm(projected.rowRange(0, 2), lastPt2D);
    std::cout << residual << endl;
}#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

// Solve system of linear equations using least squares normal equations
Eigen::MatrixXf solveLeastSquares(const Eigen::MatrixXf& pts2d,
                                         const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 11;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b(rows, 1);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z;
        b.row(i) << x;
        b.row(i + 1) << y;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Append a 1 to the end since scale is constant
    sol.conservativeResize(sol.rows() + 1, sol.cols());
    sol(sol.rows() - 1, 0) = 1;
    return sol;
}

// Overload of the above function using cv::Mat instead of Eigen

cv::Mat solveLeastSquares(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Wrap OpenCV matrices in Eigen::Map
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveLeastSquares(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}

// Solve system of linear equations using singular value decomposition
Eigen::MatrixXf solveSVD(const Eigen::MatrixXf& pts2d, const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 12;
    Eigen::MatrixXf A(rows, cols);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z, -x;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y;
    }

    // Compute the orthogonal matrix of eigenvectors of A_T*A
    Eigen::MatrixXf eigenvectors = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    // Get the eigenvector with the smallest eigenvalue (the last column of the V matrix)
    Eigen::MatrixXf smallest = eigenvectors.col(eigenvectors.cols() - 1);
    // std::cout << "sol = \n" << smallest << std::endl;
    return smallest;
}

// Overload of the above function using cv::Mat instead of Eigen
cv::Mat solveSVD(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Convert OpenCV matrices to Eigen matrices
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveSVD(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}
// Project a set of points from 3D space to the 2D image plane
cv::Mat project3D(const cv::Mat& projMat, const cv::Mat& pt3d) {
    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * pt3d;
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }

    return projected;
}

void runProblem1a() {
    cv::Mat _pts3DNorm = (cv::Mat_<float>(6,3) <<
            1.5706,-0.1490,0.2598,
            -1.5282, 0.9695, 0.3802,
            -0.6821, 1.2856, 0.4078,
            0.4124, -1.0201, -0.0915,
            1.2095, 0.2812, -0.1280,
            0.8819, -0.8481, 0.5255);

    cv::Mat _picANorm = (cv::Mat_<float>(6,2) <<
                                              1.0486, -0.3645,
                                              -1.6851, -0.4004,
                                              -0.9437, -0.4200,
                                              1.0682, 0.0699,
                                              0.6077, -0.0771,
                                              1.2543, -0.6454);

    cv::Mat _picA, _picB, _pts3D;

    // Get the last 3D normalized point so we can check our m matrix later
    cv::Mat lastPt3D = _pts3DNorm.col(_pts3DNorm.cols - 1);
    lastPt3D.push_back(1.f);
    cv::Mat lastPt2D = _picANorm.col(_picANorm.cols - 1);

    // Row-vector version of the last point so that we can print it more easily
    cv::Mat lastPt3D_T;
    cv::transpose(lastPt3D, lastPt3D_T);

    // Compute projection matrix
    {
        auto sol = solveLeastSquares(_picANorm, _pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
//        logger->info("Calibration parameters (using normal least squares):\n{}\nProjected "
//                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
//                     FormattedMat(params),
//                     lastPt3D_T,
//                     projection_T,
//                     residual);
    }

    {
        auto sol = solveSVD(_picANorm, _pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
//        logger->info("Calibration parameters (using singu    Config config(CONFIG_FILE_PATH);
//lar value decomposition):\n{}\nProjected "
//                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
//                     FormattedMat(params),
//                     lastPt3D_T,
//                     projection_T,
//                     residual);
    }
}

using namespace std;

int main() {
    Eigen::MatrixXf pts2d(6,2), pts3d(6,3);
    pts3d <<
        1.5706,-0.1490,0.2598,
        -1.5282, 0.9695, 0.3802,
        -0.6821, 1.2856, 0.4078,
        0.4124, -1.0201, -0.0915,
        1.2095, 0.2812, -0.1280,
        0.8819, -0.8481, 0.5255;

    pts2d <<
        1.0486, -0.3645,
        -1.6851, -0.4004,
        -0.9437, -0.4200,
        1.0682, 0.0699,
        0.6077, -0.0771,
        1.2543, -0.6454;


    std::cout <<pts3d.rows()  << "," << pts3d.cols() << std::endl;
    std::cout <<pts2d.rows()  << "," << pts2d.cols() << std::endl;

    std::cout << pts3d << endl;
    std::cout << pts2d << endl;

    assert(pts2d.rows() == pts3d.rows() && pts2d.cols() == 2 && pts3d.cols() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.rows() * 2;
    const size_t cols = 11;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b(rows, 1);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(i / 2, 0);
        float Y = pts3d(i / 2, 1);
        float Z = pts3d(i / 2, 2);
        float x = pts2d(i / 2, 0);
        float y = pts2d(i / 2, 1);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z;
        b.row(i) << x;
        b.row(i + 1) << y;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Append a 1 to the end since scale is constant
    sol.conservativeResize(sol.rows() + 1, sol.cols());
    sol(sol.rows() - 1, 0) = 1;

    cv::Mat cvSol;
    cv::eigen2cv(sol, cvSol);
    cv::Mat projMat = cvSol.reshape(0, 3);

    cv::Mat lastPt2D = pts2d.row(0);
    cv::Mat lastPt3D = pts3d.row(0);
    lastPt3D.push_back(1.f);


    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * lastPt3D;
    
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }
    
    // Compute residual
    double residual = cv::norm(projected.rowRange(0, 2), lastPt2D);
    std::cout << residual << endl;
}
