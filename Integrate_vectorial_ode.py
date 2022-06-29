from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

def time_deriv(t, psi, w, Omega, w_d):
    # H = w/2 * Z + Omega*cos(w_d*t)*X
    H = w / 2 * sigma_z + Omega * np.cos(w_d * t) * sigma_x
    return -1j * H @ psi


def propagate_in_time(psi_0, t_span, w, Omega, w_d, n_points, t_series=None):
    if t_series is not None:
        t = t_series
        assert t_span is None, "Double input not neccesraly consistent"
    else:
        t = np.linspace(t_span[0], t_span[1], n_points)

    time_deriv_env = lambda t, y: time_deriv(t, y, w, Omega, w_d)
    t_span = t_span if t_span is not None else (np.min(t_series), np.max(t_series))
    sol = solve_ivp(time_deriv_env, t_span, psi_0, dense_output=True)
    y = sol.sol(t)
    plt.plot(t, y[0, :], 'o--')
    # plt.plot(t, y[1, :], 'o--')
    return t, y


if __name__ == "__main__":
    psi_0 = np.array([1.0, 0], dtype=complex)
    t_span = (0, 30)
    w_true = np.random.random()
    t_measured, y_measured = propagate_in_time(psi_0, t_span, w=w_true, Omega=1, w_d=1, n_points=200)
    w_arr = np.linspace(0, 1, 11)
    Loss = np.zeros(len(w_arr))
    for i, w in enumerate(w_arr):
        _, y_w = propagate_in_time(psi_0, t_span=None, w=w, Omega=1, w_d=1, n_points=200, t_series=t_measured)
        Loss[i] = np.sum(np.absolute(y_w[0, :] - y_measured[0, :]) ** 2)
    i_estimate = np.argmin(Loss)
    w_estimate = w_arr[i_estimate]
    print("True w:" + str(w_true))
    print("Estimated w: " + str(w_estimate))
    plt.show()
