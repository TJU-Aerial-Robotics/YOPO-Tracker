#ifndef HGDO_H_
#define HGDO_H_

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

inline void get_dcm_from_q(Eigen::Matrix3d &dcm, const Eigen::Quaterniond &q) {
    float a = q.w();
    float b = q.x();
    float c = q.y();
    float d = q.z();
    float aSq = a*a;
    float bSq = b*b;
    float cSq = c*c;
    float dSq = d*d;
    dcm(0, 0) = aSq + bSq - cSq - dSq; 
    dcm(0, 1) = 2 * (b * c - a * d);
    dcm(0, 2) = 2 * (a * c + b * d);
    dcm(1, 0) = 2 * (b * c + a * d);
    dcm(1, 1) = aSq - bSq + cSq - dSq;
    dcm(1, 2) = 2 * (c * d - a * b);
    dcm(2, 0) = 2 * (b * d - a * c);
    dcm(2, 1) = 2 * (a * b + c * d);
    dcm(2, 2) = aSq - bSq - cSq + dSq;
}

class HGDO
{
public:
    HGDO(){};
    HGDO(double control_dt){
        control_dt_ = control_dt;
    };

    double Sat(double &Input)
    {
        double D = 1.0;
        double abs_input = abs(Input);
        if (abs_input <= D)
        {
            return Input;
        }
        else
        {
            return Input / abs_input;
        }
    }

    // 用这个,放到控制器的定时器和控制器同步,注意有些全局变量在FxTDO
    // 模型预测，这个z2_nhgdo作为北西天系，（用来算姿态的）加速度直接加z2_nhgdo作为实际的加速度
    // U_input: 上一时刻期望加速度（控制量姿态对应的）
    // vel: 当前时刻的速度
    // dis: 当前时刻干扰返回值,就是3个方向加速度
    void HGDO_ext_force_ob(const Eigen::Vector3d &U_input, const Eigen::Vector3d &vel, Eigen::Vector3d &dis)
    {
        Eigen::Vector3d dz1, dz2;
        Eigen::Vector3d U = U_input;
        // U(2) -= g;
        for (int i = 0; i < 3; i++)
        {
            dz1(i) = U(i) + z2_hgdo(i) - (alpha_1 / theta) * (z1_hgdo(i) - vel(i));
            dz2(i) = -(alpha_2 / pow(theta, 2)) * (z1_hgdo(i) - vel(i));
        }
        for (int i = 0; i < 3; i++)
        {
            z1_hgdo(i) += dz1(i) * control_dt_;
            z2_hgdo(i) += dz2(i) * control_dt_;
        }
        dis = z2_hgdo;
    }

    // 自适应干扰观测，立文讲先不用
    void AHGDO_ext_force_ob(Eigen::Vector3d &U_input, Eigen::Vector3d &vel, Eigen::Vector3d &dis)
    {
        Eigen::Vector3d dz1, dz2;
        Eigen::Vector3d U = U_input;
        // U(2) += g;
        for (int i = 0; i < 3; i++)
        {
            double ob = (z1_nhgdo(i) - vel(i)) / d_bound;
            dz1(i) = U(i) + z2_nhgdo(i) - (alpha_1 / theta_1) * (z1_nhgdo(i) - vel(i)) - (alpha_1 * d_bound * (1 / theta_2 - 1 / theta_1)) * Sat(ob);
            dz2(i) = -(alpha_2 / pow(theta_1, 2)) * (z1_nhgdo(i) - vel(i)) - (alpha_2 * d_bound * (1 / pow(theta_2, 2) - 1 / pow(theta_2, 2))) * Sat(ob);
        }
        for (int i = 0; i < 3; i++)
        {
            z1_nhgdo(i) += dz1(i) * control_dt_;
            z2_nhgdo(i) += dz2(i) * control_dt_;
        }
        dis = z2_nhgdo;
    }

private:
    double g = 9.81;
    double control_dt_{0.02};

    Eigen::Vector3d z1_hgdo = Eigen::Vector3d::Zero();
    Eigen::Vector3d z2_hgdo = Eigen::Vector3d::Zero();
    Eigen::Vector3d z1_nhgdo = Eigen::Vector3d::Zero();
    Eigen::Vector3d z2_nhgdo = Eigen::Vector3d::Zero();

    double theta = 0.2;
    double theta_1 = 0.2;
    double theta_2 = 0.5;
    double alpha_1 = 3.0;
    double alpha_2 = 2.0;
    double d_bound = 0.01;
};

#endif