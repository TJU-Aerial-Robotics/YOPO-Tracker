import numpy as np


def quintic_coeffs(pos0, vel0, acc0, pos1, vel1, acc1, Tf):
    """Coefficients of the quintic connecting two boundary PVA states over duration Tf.
    p(t) = sum_j coeff[j] * t^j. All arguments broadcast; returns (6, ...).
    """
    p0, v0, a0, p1, v1, a1, t = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (pos0, vel0, acc0, pos1, vel1, acc1, Tf)))
    t2 = t * t
    return np.stack([
        p0,
        v0,
        a0 / 2,
        (-10 * p0 - 6 * v0 * t - 1.5 * a0 * t2 + 10 * p1 - 4 * v1 * t + 0.5 * a1 * t2) / t ** 3,
        (15 * p0 + 8 * v0 * t + 1.5 * a0 * t2 - 15 * p1 + 7 * v1 * t - 1.0 * a1 * t2) / t ** 4,
        (-6 * p0 - 3 * v0 * t - 0.5 * a0 * t2 + 6 * p1 - 3 * v1 * t + 0.5 * a1 * t2) / t ** 5,
    ])


class Poly5Solver:
    """5-th order polynomial on a single axis."""

    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        self.A = quintic_coeffs(pos0, vel0, acc0, pos1, vel1, acc1, Tf)

    def get_snap(self, t):
        """Return the scalar snap at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t


class Polys5Solver:
    """N 5-th order polynomials on a single axis, used to visualize multiple trajectories.
    Tf: scalar, or an array of length N (one duration per trajectory).
    """

    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        self.T = np.broadcast_to(np.atleast_1d(np.asarray(Tf, dtype=np.float64)), (len(pos1),))
        self.A = quintic_coeffs(pos0, vel0, acc0, pos1, vel1, acc1, self.T)  # (6, N)

    def get_position(self, s):
        """Return the positions at normalized time s in [0, 1), each trajectory sampled at its own T."""
        t = np.atleast_1d(s)[np.newaxis, :] * self.T[:, np.newaxis]  # (N, M)
        return sum(self.A[j][:, np.newaxis] * t ** j for j in range(6)).flatten()


def wrap_to_pi(angle):
    """Wrap an angle into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_yaw(goal_dir, last_yaw, dt, max_yaw_rate=0.5):
    """Yaw command pointing at goal_dir, rate-limited to max_yaw_rate. Returns (yaw, yaw_rate)."""
    yaw_desired = np.arctan2(goal_dir[1], goal_dir[0])
    yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
    max_yaw_change = max_yaw_rate * np.pi * dt
    yaw_change = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)
    yaw = wrap_to_pi(last_yaw + yaw_change)
    return yaw, yaw_change / dt
