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
//                     lasresidualtPt3D_T,
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

#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Cholesky>
using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace std;

Matrix<double, 1, 4> solve_qr(const Matrix<double, 4, 3> &A,
                              const Matrix<double, 1, 3> &B){
    return A.transpose().colPivHouseholderQr().solve(B.transpose());
}

Matrix<double, 1, 4> solve_ldlt(const Matrix<double, 4, 3> &A, const Matrix<double, 1, 3> &B){
    return (A*A.transpose()).ldlt().solve(A*B.transpose());
}

void compare() {
    Matrix<double, 4, 3>  M_eig;
    M_eig << 761.544, 0, 0,
            0, 761.544, 0,
            639.5, 399.5, 1.0,
            3.762513283904080e+06, 1.824431013104484e+06, 9.837714402800992e+03;

    Matrix<double, 1, 3> pixelCoords_eig;
    pixelCoords_eig << 457, 520, 1;

    Matrix<double, 1, 4> worldCoords_eig;

    worldCoords_eig = solve_qr(M_eig, pixelCoords_eig);
    std::cout << "world coords using QR:   " << worldCoords_eig << std::endl;

    worldCoords_eig = solve_ldlt(M_eig, pixelCoords_eig);
    std::cout << "world coords using LDLT: " << worldCoords_eig << std::endl;
}

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
    size_t cols = 11;
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

    cols = 12;
    Eigen::MatrixXf ASVD(rows, cols);
    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(i / 2, 0);
        float Y = pts3d(i / 2, 1);
        float Z = pts3d(i / 2, 2);
        float x = pts2d(i / 2, 0);
        float y = pts2d(i / 2, 1);
        ASVD.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z, -x;
        ASVD.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y;
    }

    // Compute the orthogonal matrix of eigenvectors of A_T*A
    Eigen::MatrixXf eigenvectors = ASVD.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    std::cout << "eigenvectors:" << eigenvectors << endl;
    std::cout << "eigenvectors size:" << eigenvectors.size()  << endl;

    // Get the eigenvector with the smallest eigenvalue (the last column of the V matrix)
    Eigen::MatrixXf smallest = eigenvectors.col(eigenvectors.cols() - 1);
    std::cout << "smallest:" << smallest << endl;
    std::cout << "smallest size:" << smallest.size()  << endl;

    cv::Mat cvSol;
    cv::eigen2cv(smallest, cvSol);
    std::cout << "cvSol:" << cvSol << endl;
    std::cout << "cvSol size:" << cvSol.size()  << endl;

    cv::Mat projectionMatrixSVD = cvSol.reshape(0, 3);
    std::cout << "projectionMatrixSVD:" << projectionMatrixSVD << endl;
    std::cout << "projectionMatrixSVD size:" << projectionMatrixSVD.size()  << endl;

    // Solve least squares
    Eigen::MatrixXf leastsquare = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    std::cout << "leastsquare:" << leastsquare << endl;
    std::cout << "leastsquare size:" << leastsquare.size()  << endl;

    // Append a 1 to the end since scale is constant
    leastsquare.conservativeResize(leastsquare.rows() + 1, leastsquare.cols());
    leastsquare(leastsquare.rows() - 1, 0) = 1;
    std::cout << "leastsquare resized:" << leastsquare << endl;
    std::cout << leastsquare.size()  << endl;

    cv::Mat leaseSquareCV;
    cv::eigen2cv(leastsquare, leaseSquareCV);
    cv::Mat projectionMatrix = leaseSquareCV.reshape(0, 3);
    std::cout << "projectionMatrix:" << projectionMatrix << endl;
    std::cout << projectionMatrix.size()  << endl;


    // Find center of camera
    cv::Mat Q = projectionMatrix.colRange(0, 3);
    std::cout << "Q:" << Q << endl;
    std::cout << Q.size()  << endl;

    cv::Mat invQ = Q.inv();
    std::cout << "invQ:" << invQ << endl;
    std::cout << invQ.size()  << endl;

    cv::Mat invQnegative = -1.f * invQ;
    std::cout << "invQnegative:" << invQnegative << endl;

    cv::Mat m4 = projectionMatrix.col(3);
    std::cout << "m4:" << m4 << endl;

    cv::Mat cameraCenter = invQnegative * m4;
    std::cout << "cameraCenter:" << cameraCenter << endl;


    cv::Mat cv3dpts, cv2dpts;
    cv::eigen2cv(pts3d, cv3dpts);
    cv::eigen2cv(pts2d, cv2dpts);

    cv::Mat lastPt2D = cv2dpts.row(0);
    cv::Mat lastPt3D = cv3dpts.row(0);
    std::cout << "lastPt2D:" << lastPt2D << endl;
    std::cout << "lastPt3D:" << lastPt3D << endl;

    cv::Mat homogenousPt3D;
    homogenousPt3D.push_back(lastPt3D.at<float>(0,0));
    homogenousPt3D.push_back(lastPt3D.at<float>(0,1));
    homogenousPt3D.push_back(lastPt3D.at<float>(0,2));
    homogenousPt3D.push_back(1.f);
    std::cout << "homogenousPt3D:" << homogenousPt3D << endl;


    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projectionMatrix * homogenousPt3D;
    std::cout << "projected 2D calculated:" << projected << endl;
    cv::Mat projectedSVD = projectionMatrixSVD * homogenousPt3D;
    std::cout << "projectedSVD 2D calculated:" << projectedSVD << endl;

    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }
    std::cout << "projected 2D normalized:" << projected << endl;

    for (size_t col = 0; col < projectedSVD.cols; col++) {
        float s = projectedSVD.at<float>(2, col);
        projectedSVD.col(col) = projectedSVD.col(col) / s;
    }
    std::cout << "projectedSVD 2D normalized:" << projectedSVD << endl;


    cv::Mat projected2d = projected.rowRange(0, 2);
    std::cout << "projected2d:" << projected2d << endl;
    std::cout << projected2d.size()  << endl;

    cv::Mat projected2dSVD = projectedSVD.rowRange(0, 2);
    std::cout << "projected2dSVD:" << projected2dSVD << endl;
    std::cout << projected2dSVD.size()  << endl;


    cv::Mat transposed2D =  lastPt2D.t();
    std::cout << "sample lastPt2D:" << transposed2D << endl;
    std::cout << "sample lastPt2D size:" << transposed2D.size()  << endl;

    // Compute residualGPS
    double residual = cv::norm(projected2d, transposed2D);
    std::cout << "residual:" << residual << endl;

    double residualSVD = cv::norm(projected2dSVD, transposed2D);
    std::cout << "residualSVD:" << residualSVD << endl;


    compare();

    {
        MatrixXf cov = MatrixXf::Random(4200, 4200);
        cov = (cov + cov.transpose()) + 1000 * MatrixXf::Identity(4200, 4200);
        VectorXf b = VectorXf::Random(4200), r1, r2;

        r1 = b;
        LLT<MatrixXf> llt;
        auto start = high_resolution_clock::now();
        llt.compute(cov);
        if (llt.info() != Success)
        {
            cout << "Error on LLT!" << endl;
            return 1;
        }
        auto middle = high_resolution_clock::now();
        llt.solveInPlace(r1);
        auto stop = high_resolution_clock::now();
        cout << "LLT decomposition & solving in  " << duration_cast<milliseconds>(middle - start).count()
             << " + " << duration_cast<milliseconds>(stop - middle).count() << " ms." << endl;

        r2 = b;
        LDLT<MatrixXf> ldlt;
        start = high_resolution_clock::now();
        ldlt.compute(cov);
        if (ldlt.info() != Success)
        {
            cout << "Error on LDLT!" << endl;
            return 1;
        }
        middle = high_resolution_clock::now();
        ldlt.solveInPlace(r2);
        stop = high_resolution_clock::now();
        cout << "LDLT decomposition & solving in " << duration_cast<milliseconds>(stop - start).count()
             << " + " << duration_cast<milliseconds>(stop - middle).count() << " ms." << endl;

        cout << "Total result difference: " << (r2 - r1).cwiseAbs().sum() << endl;

    }
    return 0;
}
