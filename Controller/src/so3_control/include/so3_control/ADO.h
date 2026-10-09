#ifndef ADO_H_
#define ADO_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Core>

// 代数干扰观测器 (Algebraic Disturbance Observer), 来自 Fliess & Join 的 model-free control:
// 超局部模型 dy/dt = F + alpha*u, 这里 y 为速度, u 为期望加速度, F 即干扰加速度 (与 HGDO 的 z2 同义)
// 滑动窗口 [t-T, t] 上的代数估计:
//   F_est(t) = -6/T^3 * ∫_0^T [ (T-2σ) y(t-T+σ) + alpha (T-σ)σ u(t-T+σ) ] dσ
// 相比 HGDO: 有限时间收敛 (常值干扰恰好 T 秒后精确收敛, 无超调), 无需调极点, 旧数据 T 秒后自动遗忘
// 代价: 对时变干扰有约 T/2 的滞后, T 越小越快但越吵
class ADO
{
public:
    ADO() : ADO(0.02){};
    ADO(double control_dt, double window_T = 0.3, double alpha = 1.0)
    {
        control_dt_ = control_dt;
        alpha_ = alpha;
        N_ = std::max(2, (int)std::lround(window_T / control_dt_));
        T_ = N_ * control_dt_;
        computeWeights();
        y_buf_.assign(N_ + 1, Eigen::Vector3d::Zero());
        u_buf_.assign(N_ + 1, Eigen::Vector3d::Zero());
    };

    // 调用方式同 HGDO_ext_force_ob, 每个控制周期调用一次
    // U_input: 上一时刻期望加速度 (作用在上一周期到本时刻之间, 零阶保持)
    // vel: 当前时刻的速度
    // dis: 当前时刻干扰返回值, 就是3个方向加速度; 窗口未填满前返回 0
    void ADO_ext_force_ob(const Eigen::Vector3d &U_input, const Eigen::Vector3d &vel, Eigen::Vector3d &dis)
    {
        head_ = (head_ + 1) % (N_ + 1);
        y_buf_[head_] = vel;
        u_buf_[head_] = U_input;
        if (count_ <= N_)
            count_++;
        if (count_ <= N_)
        {
            dis.setZero();
            return;
        }

        // i = 0 对应窗口最旧的样本 (t-T), i = N 对应当前时刻
        Eigen::Vector3d F = Eigen::Vector3d::Zero();
        for (int i = 0; i <= N_; i++)
        {
            int idx = (head_ + 1 + i) % (N_ + 1);
            F += c_[i] * y_buf_[idx] + alpha_ * d_[i] * u_buf_[idx];
        }
        dis = F;
    }

private:
    double control_dt_{0.02};
    double alpha_{1.0};
    double T_{0.3};
    int N_{15};

    std::vector<double> c_, d_;
    std::vector<Eigen::Vector3d> y_buf_, u_buf_;
    int head_{0};
    int count_{0};

    // 离散化: 速度在采样间线性 (加速度零阶保持时恰好成立), 输入零阶保持, 各段积分精确求出,
    // 因此常值干扰下离散估计也是精确的
    void computeWeights()
    {
        c_.assign(N_ + 1, 0.0);
        d_.assign(N_ + 1, 0.0);
        double h = control_dt_;
        auto w = [&](double s) { return T_ - 2 * s; };
        auto P = [&](double s) { return T_ * s * s / 2 - s * s * s / 3; };  // ∫(T-σ)σ dσ 的原函数
        for (int i = 0; i < N_; i++)
        {
            double a = i * h, b = (i + 1) * h;
            c_[i] += h / 6 * (2 * w(a) + w(b));
            c_[i + 1] += h / 6 * (w(a) + 2 * w(b));
            d_[i + 1] += P(b) - P(a);  // 区间 (σ_i, σ_{i+1}] 上的输入存在 i+1 位
        }
        double k = -6.0 / (T_ * T_ * T_);
        for (int i = 0; i <= N_; i++)
        {
            c_[i] *= k;
            d_[i] *= k;
        }
    }
};

#endif
