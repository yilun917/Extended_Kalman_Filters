#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::sin;
using std::cos;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;  //store the time stamp

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  //Set the process and measurement noises
  H_laser_ << 1, 0, 0, 0,  //measurement matrix for laser
              0, 1, 0, 0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * Need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    VectorXd x_(4);
    MatrixXd P_(4, 4);
    MatrixXd Q_(4, 4);// set the acceleration noise components
    MatrixXd F_(4, 4);  //state transition matrix
    x_ << 1, 1, 1, 1; //initialize state matrix
    P_ << 1, 0, 0, 0,  //initialize coveriance matrix
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    Q_ << 0.0, 0.0, 0.0, 0.0,  //initialize Process Noise
          0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0;
    F_ << 1, 0, 1, 0,  //initialize laser state transition matrix
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;
    ekf_.Init(x_, P_, F_, H_laser_, Q_, R_laser_, R_radar_);  //initilize kalman filter
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      cout << "First data is from Radar." << endl;
      double rho = measurement_pack.raw_measurements_(0);  //store the measurement data locally
      double phi = measurement_pack.raw_measurements_(1);
      double rho_dot = measurement_pack.raw_measurements_(2);
      double local_x = rho*cos(phi);
      double local_y = rho*sin(phi);
      ekf_.x_ << local_x, local_y, rho_dot*cos(phi), rho_dot*sin(phi);  //convert polar to cartision 
    }
    else{ //if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state for laser.
      cout << "First data is from Lidar." << endl;
      double local_x = measurement_pack.raw_measurements_(0);
      double local_y = measurement_pack.raw_measurements_(1);
      ekf_.x_ << local_x, local_y, 0, 0;  //store the initial state
    }
    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;  //store initial time stamp
    is_initialized_ = true;  //set falg to initialized
    return;
  }

  /**
   * Prediction
   *
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  double dt = (measurement_pack.timestamp_  - previous_timestamp_)/ 1000000.0; //calculate the time interval
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
  double noise_ax = 9.0;
  double noise_ay = 9.0;
  
  double dt_2 = dt * dt;   // calculate powers, avoid repeat calculation
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  
  ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  ekf_.Predict();
  
  /**
   * Update
   *
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
