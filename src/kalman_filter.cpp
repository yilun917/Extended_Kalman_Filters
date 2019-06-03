#include "kalman_filter.h"
#include <cmath>
#include "tools.h"
#include <iostream>
#define _USE_MATH_DEFINES
using std::atan;
using std::sqrt;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

  
/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &Q_in, Eigen::MatrixXd &R_laser_in, Eigen::MatrixXd &R_radar_in){
  x_ = x_in;  //state matrix
  P_ = P_in;  //covariance matrix
  F_ = F_in;  //state transformation
  H_ = H_in;  //measurement function
  Q_ = Q_in;  //process noise
  R_laser = R_laser_in;  //measurement erro for laser and radar
  R_radar = R_radar_in;
  int x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);  //identity matrix
  Ht_ = H_.transpose();  //transpose for measurement matrix
}

void KalmanFilter::Predict() {
  //predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;   //state error
  MatrixXd S = H_ * P_ * Ht_ + R_laser;
  MatrixXd K = P_ * Ht_ * S.inverse();  
  x_ = x_ + K * y;   //update state
  P_ = (I_ - K * H_) * P_;   //update covariance 
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * Update the state by using Extended Kalman Filter equations
   */
  // erro checking to avoid dividing zero
  VectorXd this_z = z;  //store a local measurement 
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
  // if x and y state are small, then set rho_dot and phi to zero 
  if(px < 0.001 && py < 0.001){
    this_z(1) = 0;
    this_z(2) = 0;
  }
  
  MatrixXd Hj_(3, 4); // Jacobian matrix
  Tools tool;
  Hj_ = tool.CalculateJacobian(x_);
  MatrixXd Hjt_ = Hj_.transpose();  //  transpose of Jacbian matrix
  
  double px_2 = px * px; //calculate square, avoid repeat calculation
  double py_2 = py * py; 
  double sqrt_px2_py2 = sqrt(px_2+py_2);
  
  VectorXd hx(3);
  hx << sqrt_px2_py2, atan2(py, px), (px*vx + py*vy)/sqrt_px2_py2;  //nonlinear matrix
  
  VectorXd y = this_z - hx;
  // adjust the phi value to between -pi and pi
  while (y(1) < -M_PI){
    y(1) += 2 * M_PI;
  }
  while (y(1) > M_PI){
    y(1) -= 2 * M_PI;
  }
  
  MatrixXd S = Hj_ * P_ *Hjt_ + R_radar;
  MatrixXd K = P_ * Hjt_ * S.inverse();
  x_ = x_ + K * y;   //update state
  P_ = (I_ - K * Hj_) * P_;  //update covariance
}
