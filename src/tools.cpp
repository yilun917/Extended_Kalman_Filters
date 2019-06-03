#include "tools.h"
#include <iostream>
#include<cmath>
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::sqrt;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //Calculate the RMSE here.
  VectorXd rmse(4);
  rmse << 0,0,0,0;  //root mean square
  if(estimations.size()==0 || estimations.size()!=ground_truth.size()){
    cout<<"Estimation has error!"<<endl;  //error check for estimation not matching ground truth
    return rmse;
  }
  //calculate sum
  for (unsigned int i=0; i < estimations.size(); ++i) {
    VectorXd residue = estimations[i]-ground_truth[i];
    residue = residue.array()*residue.array();
    rmse += residue;
  }
  //calculate mean
  rmse = rmse/estimations.size();
  //calculate square root
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  //Calculate a Jacobian here.
  MatrixXd Hj_(3,4); 
  double px = x_state(0);  //locally store the state variables
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  
  double px_2 = px * px; //reduce amount of calculation
  double py_2 = py * py;
  double sqrt_px2py2_3 = sqrt((px_2+py_2) * (px_2+py_2) * (px_2+py_2));
  double sqrt_px2_py2 = sqrt(px_2+py_2);
  
  // avoid deviding zero
  if(px < 0.001 && py < 0.001){
    px = 0.001;
    py = 0.001;
  }
  
  //calculate Jacobian matrix
  Hj_ << px/sqrt_px2_py2, py/sqrt_px2_py2, 0, 0,
      -py/(px_2+py_2), px/(px_2+py_2), 0, 0,
      py*(vx*py-vy*px)/sqrt_px2py2_3, px*(vy*px-vx*py)/sqrt_px2py2_3, px/sqrt_px2_py2, py/sqrt_px2_py2;
  return Hj_;
}
