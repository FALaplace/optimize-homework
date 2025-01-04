import threading
import numpy as np
from numpy.linalg import norm
from numpy import cos
from scipy.linalg import expm
import json
import cvxpy as cp
from motion_model import RelativeMotionModel_J2

import time

Pos_obs = np.array([[-800, -500, -500],
                    [-1000, 800, 750.],
                    [150.0, 2200, 480]])

rho_obs = np.array([300, 300, 300])

d2r = lambda x: x * np.pi / 180
r2d = lambda x: x * 180 / np.pi

mu = 398600.4418 * 10 ** 9
Re = 6371
t_sam = 60

a, e, inc, w, RAAN, TA = (Re + 600) * 1000, 0.005, d2r(60), d2r(45), d2r(50.2), d2r(30)
r = a * (1 - e ** 2) / (1 + e * cos(TA))
H = np.sqrt(a * (1 - e ** 2) * mu)
vx = np.sqrt(2 * mu / r - mu / a - H ** 2 / r ** 2)
the = w + TA
c_ini = np.array([r, vx, H, the, inc, RAAN])
mm = RelativeMotionModel_J2()


def tp_fun(x):
    if x >= 10000:
        return 3.0
    if x >= 1000:
        return 1.0
    if x >= 100:
        return 0.1
    if x >= 30:
        return 0.05
    # return 0.01
    return 0.015


def cal_reference_state(U: np.ndarray, S: np.ndarray, Cr: np.ndarray, s0: np.ndarray):
    Ns, Tk = U.shape[0], U.shape[1] // 3
    Sr = np.zeros_like(S_ref)
    Sr[:, :6] = s0
    tau_r = 0.0
    for k in range(1, Tk + 1):
        for i in range(Ns):
            Jr = mm.Jacboi_matrix(Sr[i, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k])
            er = mm.ds_fsat(Sr[i, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k]) - Jr @ Sr[i, 6 * (k - 1): 6 * k]
            Ar = expm(t_sam * Jr)
            Br = t_sam * (expm(t_sam * Jr) @ mm.B + 4 * expm(t_sam * Jr / 2) @ mm.B + mm.B) / 6
            er = t_sam * (expm(t_sam * Jr) @ er + 4 * expm(t_sam * Jr / 2) @ er + er) / 6
            Sr[i, 6 * k: 6 * (k + 1)] = Ar @ Sr[i, 6 * (k - 1): 6 * k] + Br @ U[i, 3 * (k - 1): 3 * k] + er
            delta_S = norm(Sr[i, 6 * k: 6 * (k + 1)] - S[i, 6 * k: 6 * (k + 1)], ord=np.inf)
            if delta_S >= tau_r:
                tau_r = delta_S
    return Sr, tau_r


def sequential_convex_program_decoupled(Sr: np.ndarray, Cr: np.ndarray, s0: np.ndarray, st: np.ndarray, j, Ns, Tk,
                                        dict_params):
    scale = dict_params["scale"] if "scale" in dict_params else np.inf
    tau = dict_params["tau"] if "tau" in dict_params else np.inf
    umax = dict_params["umax"] if "umax" in dict_params else np.inf
    dmin = dict_params["dmin"] if "dmin" in dict_params else 0.0
    dmax = dict_params["dmax"] if "dmax" in dict_params else np.inf
    pos_obs = dict_params["obstacle_position"] if "obstacle_position" in dict_params else None
    r_obs = dict_params["obstacle_radius"] if "obstacle_radius" in dict_params else None
    c_pun = dict_params["collision_punish"] if "collision_punish" in dict_params else 3.0
    t_pun = dict_params["terminal_punish"] if "terminal_punish" in dict_params else 3.0
    r_pun = dict_params["region_punish"] if "region_punish" in dict_params else 0.5

    Uj = cp.Variable(Tk * 3)
    Sj = dict.fromkeys(range(Tk + 1))
    Sj[0] = s0[j]
    constraints = []
    objective = 0.0
    for k in range(1, Tk + 1):
        Jac = mm.Jacboi_matrix(Sr[j, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k])
        err = mm.ds_fsat(Sr[j, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k]) - Jac @ Sr[j, 6 * (k - 1): 6 * k]
        Ad = expm(t_sam * Jac)
        Bd = t_sam * (expm(t_sam * Jac) @ mm.B + 4 * expm(t_sam * Jac / 2) @ mm.B + mm.B) / 6
        erd = t_sam * (expm(t_sam * Jac) @ err + 4 * expm(t_sam * Jac / 2) @ err + err) / 6
        Sj[k] = Ad @ Sj[k - 1] + Bd @ Uj[3 * (k - 1): 3 * k] + erd
        objective += cp.norm(Uj[3 * (k - 1): 3 * k]) * t_sam  # todo! energy saving performance
        objective += cp.maximum(0.0, cp.norm(Sj[k][:3]) - scale) * r_pun  # todo! region punishment
        constraints.append(cp.norm(Sj[k] - Sr[j, 6 * k: 6 * (k + 1)], p="inf") <= tau)  # todo! trust region
        constraints.append(cp.norm(Uj[3 * (k - 1): 3 * k]) <= umax)  # todo! max thrust constraint
        # todo!!! convexification for non-convex constraints: soft constraints
        for m in range(Ns):
            if m != j:
                Ef = Sr[j, 6 * k: 6 * k + 3] - Sr[m, 6 * k: 6 * k + 3]
                if norm(Ef) < dmin + 2.0 * tau:
                    objective += cp.maximum(
                        (dmin + 0) * norm(Ef) - Ef @ (Sj[k][:3] - Sr[m, 6 * k: 6 * k + 3]), 0.0) * c_pun / 2
                constraints.append(cp.norm(Sj[k][:3] - Sr[m, 6 * k: 6 * k + 3]) <= (dmax - 0))
        if pos_obs is not None:
            for o in range(r_obs.size):
                po, ro = pos_obs[o], r_obs[o]
                Df = Sr[j, 6 * k: 6 * k + 3] - po
                if norm(Df) < ro + tau:
                    objective += cp.maximum(ro * norm(Df) - Df @ (Sj[k][:3] - po), 0.0) * c_pun
    # todo! terminal state constraints
    objective += cp.norm(Sj[Tk] - st) * t_pun
    opt = cp.Minimize(objective)
    prob = cp.Problem(opt, constraints)
    prob.solve(solver=cp.MOSEK)
    return Uj.value, prob.value


def thread_fun(Sr, Cr, s0: np.ndarray, st: np.ndarray, j, Ns, Tk, res, param_dict):
    result = sequential_convex_program_decoupled(Sr, Cr, s0, st, j, Ns, Tk, param_dict)
    res[j] = result


if __name__ == "__main__":
    keys = ["X", "U", "initial", "target", "umax", "scale", "dmax", "dmin"]

    with open("data/initial_data.json", "r") as f:
        ini_state_recorder = json.load(f)
    U_ref = np.array(ini_state_recorder["U"])
    N, K = U_ref.shape[0], U_ref.shape[1] // 3
    S_ref = np.empty((N, 6 * (K + 1)))
    C_ref = np.empty(6 * (K + 1))
    s_ini = np.array(ini_state_recorder["initial"])
    s_tar = np.array(ini_state_recorder["target"])
    S_ref[:, :6] = s_ini
    C_ref[0:6] = c_ini
    ck = c_ini
    for k in range(1, K + 1):
        for i in range(N):
            dsi_k = mm.ds_fsat(S_ref[i, 6 * (k - 1): 6 * k], ck) + mm.B @ U_ref[i, 3 * (k - 1): 3 * k]
            S_ref[i, 6 * k: 6 * (k + 1)] = S_ref[i, 6 * (k - 1): 6 * k] + t_sam * dsi_k
        ck = ck + t_sam * mm.derivation_csat(ck)
        C_ref[6 * k: 6 * (k + 1)] = ck

    tau = 50.0
    obj = np.inf
    counter = 0
    start_time = time.time()
    file = open('data/iteration_log.txt', 'a')
    file.write("the Trajectory planning has been started Algorithm: d-CSCP")
    while True:
        outputs = [None] * N
        threads = []
        kwargs = {
            "scale": ini_state_recorder["scale"],
            "umax": ini_state_recorder["umax"],
            "dmax": ini_state_recorder["dmax"],
            "dmin": ini_state_recorder["dmin"],
            "tau": tau,
            "obstacle_position": Pos_obs,
            "obstacle_radius": rho_obs,
            "terminal_punish": tp_fun(obj),
            "collision_punish": 3.0,
            "region_punish": 0.5,
        }
        for j in range(N):
            thread = threading.Thread(target=thread_fun, args=(S_ref, C_ref, s_ini, s_tar, j, N, K, outputs, kwargs))
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        outputs_T = list(zip(*outputs))
        obj_ = sum(outputs_T[1])
        U_ = np.array(outputs_T[0])
        counter += 1
        S_, tau_real = cal_reference_state(U_, S_ref, C_ref, s_ini)
        file.write(f'The {counter} iteration, the cost is {obj_}\n')
        print(counter, obj_, tau_real)
        U_ref = U_.copy()
        S_ref = S_.copy()
        tau = tau_real * 0.99
        if tau_real <= 1.0 or abs(obj_ - obj) <= 0.01:
            break
        obj = obj_
    end_time = time.time()
    # print(end_time - start_time)
    np.save("data/optimized_ctrl_decoupled_SCP.npy", U_ref)
    np.save("data/optimized_state_decoupled_SCP.npy", S_ref)
    with open('data/iteration_log_de_3.txt', 'a') as file:
        file.write(f'The sequential convex programming takes {int(end_time - start_time)} seconds\n')
