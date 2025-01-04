import numpy as np
import cvxpy as cp
from motion_model import RelativeMotionModel
from numpy.linalg import matrix_power as mp
from numpy.linalg import norm
import json
import time

mu = 398600.4418
Re = 6371
t_sam = 60.0  # the magnitude of time interval
N = 3  # the scale of spacecraft formation
K = 100  # the total time step
u_max = 3.5e-3  # the maximum force
d_max = 300  # the maximum distance between each satellite pair
d_min = 50.  # the minimum distance between each satellite pair

np.random.seed(42)
rho = 2000
pc = np.random.random((3,)) * np.random.choice([-1, 1], size=3)
pc = pc / norm(pc) * rho
vc = np.random.random((3,)) * np.random.choice([-1, 1], size=3)
tc = np.array([-pc[0], -pc[1], 0.0, 0.0, 0.0, 0.0])


def convert_np_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError


def objective_fun(u: cp.Variable, x_ini, x_tar):
    """
    :param u: shape = [Ns, Tk * 3]
    :param x_ini: initial state of spacecraft
    :param x_tar: target state of spacecraft
    :return:
    """
    scale = rho * 1.1
    Q = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0])
    Ns = u.shape[0]
    Tk = u.shape[1] // 3
    J = 0.0
    # x0 = x_ini
    # xf = x_tar
    # todo!!! for each time interval loop
    for k in range(1, Tk + 1):
        Wjk = np.concatenate([mp(MM.Ad, k - ka) @ MM.Bd for ka in range(1, k + 1)], axis=1)
        Ajk = mp(MM.Ad, k)
        # todo!!! for each spacecraft loop
        for j in range(Ns):
            xjk = Ajk @ x_ini[j, :] + Wjk @ u[j, :3 * k]
            J += cp.maximum(0.0, cp.norm(xjk[:3]) - scale) * 0.5 + cp.norm(u[j, 3 * (k - 1):3 * k]) * t_sam
            # if k == Tk:
            #     # todo!!! terminal state performance
            #     J += cp.norm(Q @ (xjk - x_tar)) * 0.001
        pass
    return J


if __name__ == "__main__":
    """(Linear) motion model parameters"""
    r = Re + 600
    w = np.sqrt(mu / r ** 3)
    MM = RelativeMotionModel(w)
    """initial state of each spacecraft"""
    p_ini = np.random.random((N, 3)) * np.random.choice([-1, 1], size=3)
    v_ini = np.random.random((N, 3)) * np.random.choice([-1, 1], size=3)
    p_ini = p_ini / norm(p_ini, axis=1) * 100 + pc
    v_ini = v_ini / norm(v_ini, axis=1) * 0.1 + vc
    s_ini = np.concatenate((p_ini, v_ini), axis=1)
    """optimize setting"""
    start_time = time.time()
    U = cp.Variable((N, 3 * K))
    constraints = []
    for k in range(1, K + 1):
        Wk = np.concatenate([mp(MM.Ad, k - ka) @ MM.Bd for ka in range(1, k + 1)], axis=1)
        Ak = mp(MM.Ad, k)
        sk_dict = dict.fromkeys(range(N))
        for i in range(N):
            # todo!!! maximum control force constraints
            constraints.append(cp.norm(U[i, 3 * (k - 1):3 * k], 2) <= u_max)
            sik = Ak @ s_ini[i] + Wk @ U[i, :3 * k]
            sk_dict[i] = sik
            for j in range(i):
                sjk = sk_dict[j]
                # todo!!! maximum communication distance constraints
                constraints.append(cp.norm(sik[:3] - sjk[:3]) <= d_max)
            if k == K:
                constraints.append(cp.norm(sik[:3] - tc[:3]) <= 1.0)
    objective = cp.Minimize(objective_fun(U, s_ini, tc))
    prob1 = cp.Problem(objective, constraints)
    prob1.solve(cp.MOSEK)
    end_time = time.time()
    print(prob1.status, prob1.value, end_time - start_time)
    U_seq: np.ndarray = U.value
    S_seq = np.zeros((N, 6 * (K + 1)))
    S_seq[:, :6] = s_ini
    for i in range(N):
        for k in range(1, K + 1):
            S_seq[i, 6 * k: 6 * (k + 1)] = MM.Ad @ S_seq[i, 6 * (k - 1): 6 * k] + MM.Bd @ U_seq[i, 3 * (k - 1): 3 * k]

    from Visualization import track_visualization

    tv = track_visualization()
    tv.plot_fsat_track(S_seq.reshape((N, K + 1, 6)), tc)

    # recorder = {"X": S_seq, "U": U.value, "initial": s_ini, "target": tc, "umax": u_max,
    #             "scale": rho * 1.1, "dmax": d_max, "dmin": d_min}
    # with open("data/initial_data.json", "w") as f:
    #     json.dump(recorder, f, default=convert_np_array)
    # print("data has been recorded.")
