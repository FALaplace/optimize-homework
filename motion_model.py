import numpy as np
from numpy import sin, cos
from scipy.linalg import expm
from scipy.optimize import approx_fprime

d2r = lambda x: x * np.pi / 180
r2d = lambda x: x * 180 / np.pi
Re = 6371.0 * 1000
mu = 398600.4418 * 10 ** 9
J2 = 1.08264 * 10 ** (-3)
k_J2 = 3 * J2 * mu * Re ** 2 / 2


# k_J2 = 0.0

class RelativeMotionModel:
    def __init__(self, w, T_sam=60):
        self.w = w
        self.T_sam = T_sam
        self.Period = 2 * np.pi / w
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [3 * w ** 2, 0, 0, 0, 2 * w, 0],
                           [0, 0, 0, -2 * w, 0, 0],
                           [0, 0, -w ** 2, 0, 0, 0]])
        self.B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Ad = expm(T_sam * self.A)  # discrete system matrix
        self.Bd = T_sam * (expm(T_sam * self.A) @ self.B + 4 * expm(T_sam * self.A / 2) @ self.B + self.B) / 6

    def render(self, s0: np.ndarray, **kwargs) -> np.ndarray:
        Tend = kwargs["Tend"] if "Tend" in kwargs.keys() else self.Period * 3.0
        # sigma_v = self.w ** 2 * 0.05
        sigma_v = 0
        s_seq = [s0.copy()]
        sk = s0
        ti = 0
        while True:
            sk = self.Ad @ sk + self.Bd @ np.random.normal(scale=sigma_v, size=(3,))
            s_seq.append(sk)
            ti += self.T_sam
            if ti >= Tend or np.linalg.norm(sk[:3]) <= 1.0:
                break
        s_seq = np.array(s_seq)
        return s_seq


class RelativeMotionModel_J2:
    """
    The center satellite state is denoted as sc; the j-th follow satellite state is denoted as sj
    sc = [r, vx, h, the, inc, RAAN]
    sj = [xj, yj, zj, vxj, vyj, vzj]
    The non-linear motion equation of satellite clusters:
    dsj = fj(sj; sc) + B @ u
    """

    def __init__(self):
        self.B = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=0)
        self.G = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

    @staticmethod
    def derivation_csat(c):
    # 从星
        r, vx, h, the, i, RAAN = c
        dr = vx
        dvx = -mu / r ** 2 + h ** 2 / r ** 3 - k_J2 * (1 - 3 * sin(i) ** 2 * sin(the) ** 2) / r ** 4
        dh = -k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 3
        dthe = h / r ** 2 + 2 * k_J2 * (cos(i) ** 2 * sin(the) ** 2) / (h * r ** 3)
        dinc = -k_J2 * (sin(2 * i) * sin(2 * the)) / (2 * h * r ** 3)
        dRAAN = -2 * k_J2 * cos(i) * sin(the) ** 2 / (h * r ** 3)
        return np.array([dr, dvx, dh, dthe, dinc, dRAAN])

    @staticmethod
    def ds_fsat(s, args: np.ndarray):
        r, vx, h, the, i, RAAN = args
        wx = -k_J2 * (sin(2 * i) * sin(the)) / (h * r ** 3)
        wz = h / r ** 2
        ax = (-k_J2 * (sin(2 * i) * cos(the)) / r ** 5 + 3 * vx * k_J2 * (sin(2 * i) * sin(the)) / (r ** 4 * h) -
              8 * k_J2 ** 2 * sin(i) ** 3 * cos(i) * sin(the) ** 2 * cos(the) / (r ** 6 * h ** 2))
        az = -2 * h * vx / r ** 3 - k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 5
        eta = mu / r ** 3 + k_J2 / r ** 5 - 5 * k_J2 * sin(i) ** 2 * sin(the) ** 2 / r ** 5
        kexi = 2 * k_J2 * (sin(i) * sin(the)) / r ** 4
        x, y, z, vx, vy, vz = s
        rf = np.sqrt((r + x) ** 2 + y ** 2 + z ** 2)
        rfz = (r + x) * sin(i) * sin(the) + y * sin(i) * cos(the) + z * cos(i)
        etaf = mu / rf ** 3 + k_J2 / rf ** 5 - 5 * k_J2 * rfz ** 2 / rf ** 7
        kexif = 2 * k_J2 * rfz / rf ** 5
        dx, dy, dz = vx, vy, vz
        dvx = (2 * vy * wz - x * (etaf - wz ** 2) + y * az - z * wx * wz -
               (kexif - kexi) * sin(i) * sin(the) - r * (etaf - eta))
        dvy = (-2 * vx * wz + 2 * vz * wx - x * az - y * (etaf - wz ** 2 - wx ** 2) + z * ax -
               (kexif - kexi) * sin(i) * cos(the))
        dvz = -2 * vy * wx - x * wx * wz - y * ax - z * (etaf - wx ** 2) - (kexif - kexi) * cos(i)
        dx = np.array([dx, dy, dz, dvx, dvy, dvz])
        return dx

    def Jacboi_matrix(self, x, args=None):
        jac = approx_fprime(x, self.ds_fsat, np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]), args)
        return jac
