import numpy as np
from Visualization import track_visualization
import json


def convert_np_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError


def format_file(init_json, opt_state_npy, opt_ctrl_npy, opt_data_json):
    with open(init_json, "r") as f:
        ini_state_recorder: dict = json.load(f)
    X_opt = np.load(opt_state_npy)
    U_opt = np.load(opt_ctrl_npy)
    Pos_obs = np.array([[-800, -500, -500],
                        [-1000, 800, 750.],
                        [150.0, 2200, 480]])

    rho_obs = np.array([300, 300, 300])

    ini_state_recorder["obstacle_position"] = Pos_obs
    ini_state_recorder["obstacle_radius"] = rho_obs
    del ini_state_recorder["X"]
    del ini_state_recorder["U"]
    ini_state_recorder["X_opt"] = X_opt
    ini_state_recorder["U_opt"] = U_opt
    print(ini_state_recorder.keys())
    with open(opt_data_json, "w") as f:
        json.dump(ini_state_recorder, f, default=convert_np_array)
    print("data has been recorded.")
    pass


def opt_test(obs_pos, obs_rad, s_opt):
    obs_num = obs_rad.size
    N = s_opt.shape[0]
    dis_so = np.empty([N, s_opt.shape[1], obs_num])

    # sat & obs
    for i in range(N):
        for j in range(obs_num):
            si = s_opt[i]
            dis = np.array([[si[:, 0] - obs_pos[j, 0]], [si[:, 1] - obs_pos[j, 1]], [si[:, 2] - obs_pos[j, 2]]])
            dis_so[i, :, j] = np.linalg.norm(dis, axis=0)
    min_dis = np.min(dis_so, axis=1)
    print("Minimal distance for each sat with each obstacle:\n", min_dis)
    for i in range(N):
        opt_num_so = 0
        for j in range(obs_num):
            if min_dis[i, j] < obs_rad[j]:
                print(f"Collision detected: sat {i}, obstacle {j}")
                break
            if np.abs(min_dis[i, j] - obs_rad[j]) < 50:  # tau
                opt_num_so += 1
        if opt_num_so == 3:
            print(f"Optimal trajectory for sat {i} with each obstacle")

    # sat & sat
    for i in range(N):
        opt_num_ss = 0
        for ii in range(i + 1, N):
            si = s_opt[i]
            sii = s_opt[ii]
            dis = np.array([[si[:, 0] - sii[:, 0]], [si[:, 1] - sii[:, 1]], [si[:, 2] - sii[:, 2]]])
            dis_ss = np.linalg.norm(dis, axis=0)
            print(np.min(dis_ss))
            if np.any(dis_ss < 50):  # dmin
                print(f"Too close: sat {i}, sat {ii}!")
                break
            else:
                opt_num_ss += 1
        if opt_num_ss == N - i - 1:
            print(f"Optimal trajectory for sat {i} with other sat")


if __name__ == "__main__":
    keys = ["X_opt", "U_opt", "initial", "target", "umax", "scale", "dmax", "dmin",
            "obstacle_position", "obstacle_radius"]

    ini_json = "data/initial_data.json"
    CSCP_json = "data/optimized_data_coupled_SCP.json"
    DCSCP_json = "data/optimized_data_decoupled_SCP.json"

    with open(DCSCP_json, "r") as f:
        data: dict = json.load(f)
    s = np.array(data["X_opt"]) if "X_opt" in data else data["X"]
    u = np.array(data["U_opt"]) if "U_opt" in data else data["U"]
    Ns, Tk = s.shape
    st = data["target"]
    obs_pos = np.array(data["obstacle_position"])
    obs_rad = np.array(data["obstacle_radius"])
    tv = track_visualization()
    tv.show_obstacle(obs_pos, obs_rad)
    tv.plot_fsat_track(s.reshape((Ns, Tk // 6, 6)), st)
    opt_test(obs_pos, obs_rad, s.reshape((Ns, Tk // 6, 6)))
