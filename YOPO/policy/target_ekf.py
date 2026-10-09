import numpy as np
from collections import deque


class EKF:
    def __init__(self, dt=0.1, distance_threshold=1, confirm_m=4, confirm_n=6, max_blind_time=0.5, target_max_speed=10.0):
        self.dt = dt
        self.distance_threshold = distance_threshold
        self.gate_growth = 6.0                    # gate growth per second of dead reckoning (m/s)
        self.gate_max = 3.0 * distance_threshold  # hard cap on the gate
        self.max_blind_time = max_blind_time      # dead-reckoning timeout, beyond which the track counts as lost
        self.blind_time = 0.0
        self.target_max_speed = target_max_speed  # target top speed, sets the re-acquisition reachable radius
        self.anchor = np.zeros(3)                 # last confirmed target position, centre of the reachable set
        self.lost_time = np.inf                   # time since that confirmation (inf before the first one)
        self.confirm_m = confirm_m
        self.hits = deque(maxlen=confirm_n)
        self.lost = True  # lost; re-acquiring needs m hits within the last n frames

        # Define matrices
        self.A = np.eye(6)
        self.B = np.zeros((6, 3))
        self.C = np.zeros((3, 6))
        self.Qt = np.eye(3)
        self.Rt = np.eye(3)
        self.Sigma = np.eye(6) * 5
        self.K = np.zeros((6, 3))
        self.x = np.zeros(6)

        # Set up A matrix
        self.A[0, 3] = dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt

        # Set up B matrix
        t2 = dt ** 2 / 2
        self.B[0, 0] = t2
        self.B[1, 1] = t2
        self.B[2, 2] = t2
        self.B[3, 0] = dt
        self.B[4, 1] = dt
        self.B[5, 2] = dt

        # Set up C matrix
        self.C[0, 0] = 1
        self.C[1, 1] = 1
        self.C[2, 2] = 1

        # Set up Qt and Rt matrices
        # Process noise (acceleration), larger allows more target manoeuvring
        self.Qt[0, 0] = 16
        self.Qt[1, 1] = 16
        self.Qt[2, 2] = 8
        # Measurement noise, smaller trusts the detections more
        self.Rt[0, 0] = 0.02
        self.Rt[1, 1] = 0.02
        self.Rt[2, 2] = 0.02

    def predict(self):
        # Freeze once dead reckoning times out; gated on blind_time, not self.lost, so a pending track
        # that is observed every frame still predicts
        if self.blind_time >= self.max_blind_time:
            return
        self.x = self.A @ self.x
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.B @ self.Qt @ self.B.T

    def reset(self, z):
        self.x[:3] = z
        self.x[3:] = 0
        self.Sigma = np.eye(6)

    def check_valid(self, z):
        """Gate the innovation ||z - x_pred|| against distance_threshold + sqrt(Sigma_pp) +
        gate_growth * blind_time, capped at gate_max. Returns (indices inside the gate, closest index).
        """
        if z.ndim == 1:
            z = z[np.newaxis, :]
        sigma_p = np.sqrt(np.trace(self.Sigma[:3, :3]) / 3.0)
        gate = min(self.distance_threshold + sigma_p + self.gate_growth * self.blind_time, self.gate_max)
        distances = np.linalg.norm(z - self.x[:3], axis=1)
        return np.where(distances < gate)[0], np.argmin(distances)

    def update(self, z):
        self.K = self.Sigma @ self.C.T @ np.linalg.inv(self.C @ self.Sigma @ self.C.T + self.Rt)
        self.x = self.x + self.K @ (z - self.C @ self.x)
        self.Sigma = self.Sigma - self.K @ self.C @ self.Sigma

    def is_prediction_reliable(self):
        return self.blind_time <= self.max_blind_time

    def pos(self):
        return self.x[:3].copy()  # a copy; the caller stores it and reset() writes self.x in place

    def vel(self):
        return self.x[3:]

    def predict_and_update(self, target_positions):
        """
        Returns: valid indices, ekf prediction
        """
        # Predict the next state
        self.predict()

        valid_ids, updated, reinit = np.empty(0, dtype=int), False, False
        if target_positions.size > 0:
            valid_ids, best_id = self.check_valid(target_positions)
            if valid_ids.size > 0:
                # If the closest target is within the threshold, update the state
                self.update(target_positions[best_id])
                updated = True
            elif self.lost:
                # Re-initialize from a detection inside the reachable set
                # (last confirmed position + v_max * time lost), otherwise stay lost
                reach = self.target_max_speed * self.lost_time + self.gate_max
                if np.linalg.norm(target_positions[best_id] - self.anchor) < reach:
                    self.reset(target_positions[best_id])
                    valid_ids, reinit = np.array([best_id]), True
        self.blind_time = 0.0 if (updated or reinit) else self.blind_time + self.dt

        if not self.is_prediction_reliable():
            self.lost = True
            self.hits.clear()
        elif self.lost:                 # re-acquisition: m hits within the last n frames
            self.hits.append(updated)  # a re-init frame is not a hit
            if sum(self.hits) >= self.confirm_m:
                self.lost = False

        if updated and not self.lost:   # only a confirmed track moves the reachable-set centre
            self.anchor, self.lost_time = self.x[:3].copy(), 0.0
        else:
            self.lost_time += self.dt
        return (np.empty(0) if self.lost else valid_ids), self.pos(), not self.lost


if __name__ == "__main__":
    dt = 0.1  # Time step
    ekf = EKF(dt)

    closest_target, ekf_predicted_pos = ekf.predict_and_update(np.array([[0.5, 0.0, 1.0]]))
    closest_target, ekf_predicted_pos = ekf.predict_and_update(np.array([[1.0, 0.0, 1.0]]))
    closest_target, ekf_predicted_pos = ekf.predict_and_update(np.array([[2.0, 0.0, 1.0]]))
    closest_target, ekf_predicted_pos = ekf.predict_and_update(np.array([[2.5, 0.0, 1.0]]))
