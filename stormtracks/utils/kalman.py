# Independent.
import numpy as np
import matplotlib.pyplot as plt


class RTSSmoother(object):
    '''Implementation of Rauch-Tung-Striebel smoother'''
    def __init__(self, F, H):
        self.kf = KalmanFilter(F, H)
        self.predicted_xs = []
        self.predicted_Ps = []
        self.filtered_xs = []
        self.filtered_Ps = []
        self.xs = []
        self.Ps = []

    def process_data(self, zs, x, P, Q, R):
        '''
        :param zs: measurements
        :param x: initial state
        :param P: initial covariance matrix
        :param Q: process uncertainty covariance matrix
        :param R: measurement uncertainty covariance matrix

        :returns: self.xs: all smoothed x values, self.Ps: all smoothed P values
        '''

        self.predicted_xs = []
        self.predicted_Ps = []
        self.filtered_xs = []
        self.filtered_Ps = []
        self.xs = []
        self.Ps = []

        self.zs = zs

        # Forward pass, calc and store filtered/predicted x/P:
        for z in zs:
            # Calc next x, P given current x, P and measurement z.
            x, P = self.kf.estimate(x, P, np.matrix([z]).T, Q, R)

            # Store result.
            self.predicted_xs.append(self.kf.x_predicted)
            self.predicted_Ps.append(self.kf.P_predicted)
            self.filtered_xs.append(x)
            self.filtered_Ps.append(P)

            # Make sure xs, Ps have correct length.
            self.xs.append(0)
            self.Ps.append(0)

        # Initialise last values of xs, Ps
        self.xs[-1] = self.filtered_xs[-1]
        self.Ps[-1] = self.filtered_Ps[-1]

        # Backward pass, calc and store smoothed x/P (skip last value due to i + 1):
        for i in range(len(self.filtered_xs) - 1)[::-1]:
            P_filtered, P_predicted = self.filtered_Ps[i], self.predicted_Ps[i]
            x_filtered, x_predicted = self.filtered_xs[i], self.predicted_xs[i]

            # Perform calc.
            C = P * self.kf.F.T * P_predicted.I
            x_smoothed = x_filtered + C * (self.xs[i + 1] - x_predicted)
            P_smoothed = P_filtered + C * (self.Ps[i + 1] - P_predicted) * C.T

            self.xs[i] = x_smoothed
            self.Ps[i] = P_smoothed

        return self.xs, self.Ps


class KalmanFilter(object):
    '''Implementation of static Kalman Filter

    Assumes a static model and observation operator.

    :param F: model to use to update x (np.matrix)
    :param H: observation operator (np.matrix)
    '''
    def __init__(self, F, H):
        self.F = F
        self.H = H
        self.I = np.matrix(np.eye(F.shape[0]))  # identity matrix

    def predict(self, x_init, P_init, Q):
        '''Predict where x, P will be based on model'''
        # import ipdb; ipdb.set_trace()
        F = self.F

        # Predict
        self.x_predicted = F * x_init
        self.P_predicted = F * P_init * F.T + Q

        return self.x_predicted, self.P_predicted

    def update(self, x, P, z, R):
        '''Update x, P based on new observation'''
        H = self.H
        I = self.I

        # Update
        # import ipdb; ipdb.set_trace()
        y = z - H * x
        S = H * P * H.T + R
        K = P * H.T * S.I
        self.x = x + K * y
        self.P = (I - K * H) * P

        self.z = z
        self.y = y
        self.S = S
        self.K = K

        return self.x, self.P

    def estimate(self, x_init, P_init, z, Q, R):
        '''Perform predict then update

        :param Q: covariance matrix for uncertainty in model
        :param R: covariance matrix for uncertainty in observation
        :returns: best estimate for x, P
        '''
        x, P = self.predict(x_init, P_init, Q)
        x, P = self.update(x, P, z, R)

        return x, P

    def _update_multiple(self, x, P, zs, R):
        for z in zs:
            x, P = self.update(x, P, np.matrix([x]).T, R)

        return x, P

    def _estimate_multiple(self, x_init, P_init, zs, Q, R):
        x, P = self.predict(x_init, P_init, Q)
        x, P = self.update_multiple(x, P, zs, R)

        return x, P


class Linear2DKalman(KalmanFilter):
    '''Linear 2D version of KalmanFilter'''
    def __init__(self):
        F = np.matrix([[1., 0., 1., 0.],
                       [0., 1., 0., 1.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        H = np.matrix([[1., 0., 0., 0.],
                       [0., 1., 0., 0.]])

        super(Linear2DKalman, self).__init__(F, H)


def _demo_voltage(zs=None):
    sz = (50,)
    x_true = -0.37727
    xs = np.ones(sz) * x_true
    if zs is None:
        zs = np.random.normal(0, 0.1, size=sz)
    obs_xs = xs + zs

    result = _demo_simple_1d(obs_xs)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, 'k+', label='noisy measurements')
    plt.plot(result, 'b-', label='a posteriori estimate')
    plt.axhline(x_true, color='g', label='truth value')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')

    return obs_xs, zs, result


def _demo_simple_1d(obs_xs, Q=1e-5, R=0.1 ** 2, show_working=False):
    F = np.matrix([1])
    H = np.matrix([1])

    kf = KalmanFilter(F, H)

    x = np.matrix([0])
    P = np.matrix([1])

    result = []
    result.append(x[0, 0])
    for obs_x in obs_xs[1:]:
        x, P = kf.estimate(x, P, np.matrix([obs_x]), Q, R)
        result.append(x[0, 0])

        if show_working:
            print('x_predicted: {0}'.format(kf.x_predicted[0, 0]))
            print('P_predicted: {0}'.format(kf.P_predicted[0, 0]))

            print('K          : {0}'.format(kf.K[0, 0]))
            print('z          : {0}'.format(kf.z[0, 0]))
            print('y          : {0}'.format(kf.y[0, 0]))

            print('x          : {0}'.format(x[0, 0]))
            print('P          : {0}'.format(P[0, 0]))

    return result


def _demo_2d_with_inertia(noise=0.5, Q=1e-2, R=0.5 ** 2):
    sz = (50, 2)
    true_xs = np.linspace(0, sz[0])
    true_xs[25:] = np.ones(25) * 25
    # true_ys = 4 * np.sin(true_xs / 5)
    true_ys = np.zeros_like(true_xs)
    true_ys[25:] = true_xs[:25]

    zs = np.random.normal(0, noise, size=sz)

    obs_xs = true_xs + zs[:, 0]
    # obs_xs = true_xs
    obs_ys = true_ys + zs[:, 1]

    x_result, P_result, y_result = _demo_simple_2d_with_inertia(zip(obs_xs, obs_ys), Q, R)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, obs_ys, 'k+', label='noisy measurements')
    plt.plot(x_result[:, 0], x_result[:, 1], 'b-', label='a posteriori estimate')

    plt.figure(2)
    plt.clf()
    plt.plot(y_result, 'b-', label='y')
    # plt.axhline(x_true ,color='g', label='truth value')
    # plt.legend()

    return zip(obs_xs, obs_ys), x_result


def _demo_2d(Q=1e-5, R=0.1 ** 2):
    sz = (50, 2)
    true_xs = np.linspace(0, sz[0])
    true_ys = true_xs ** 2

    zs = np.random.normal(0, 100, size=sz)

    # obs_xs = true_xs + zs[:, 0]
    obs_xs = true_xs
    obs_ys = true_ys + zs[:, 1]

    Q = np.matrix([[2, 1], [1, 2]]) * Q
    R = np.matrix([[2, 1], [1, 2]]) * R

    result = _demo_simple_2d(zip(obs_xs, obs_ys), Q, R)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, obs_ys, 'k+', label='noisy measurements')
    plt.plot(result[:, 0], result[:, 1], 'b-', label='a posteriori estimate')
    # plt.axhline(x_true ,color='g', label='truth value')
    # plt.legend()

    return zip(obs_xs, obs_ys), result


def _demo_simple_2d(obs_points, Q=1e-5, R=0.1 ** 2, show_working=False):
    F = np.matrix(np.eye(2))
    H = np.matrix(np.eye(2))

    kf = KalmanFilter(F, H)

    x = np.matrix([0, 0]).T
    P = np.matrix(np.eye(2))

    result = []
    result.append(x.getA()[:, 0])
    for obs_point in obs_points[1:]:
        x, P = kf.estimate(x, P, np.matrix([obs_point]).T, Q, R)
        result.append(x.getA()[:, 0])

        if show_working:
            print('x_predicted: {0}'.format(kf.x_predicted[0, 0]))
            print('P_predicted: {0}'.format(kf.P_predicted[0, 0]))

            print('K          : {0}'.format(kf.K[0, 0]))
            print('z          : {0}'.format(kf.z[0, 0]))
            print('y          : {0}'.format(kf.y[0, 0]))

            print('x          : {0}'.format(x[0, 0]))
            print('P          : {0}'.format(P[0, 0]))

    return np.array(result)


def _demo_simple_2d_with_inertia(x_init, obs_points, Q=1e-5, R=0.1 ** 2, show_working=False):
    F = np.matrix([[1., 0., 1., 0.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    H = np.matrix([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])

    Q = np.matrix([[2., 1., 1., 1.],
                   [1., 2., 1., 1.],
                   [1., 1., 2., 1.],
                   [1., 1., 1., 2.]]) * Q
    R = np.matrix([[2., 1.],
                   [1., 2.]]) * R

    kf = KalmanFilter(F, H)

    x = np.matrix([x_init[0], x_init[1], 0, 0]).T
    P = np.matrix(np.eye(4)) * 10

    result = []
    P_result = []
    y_result = []

    result.append(x.getA()[:, 0])
    P_result.append(np.linalg.det(P))

    for obs_point in obs_points[1:]:
        # import ipdb; ipdb.set_trace()
        x, P = kf.estimate(x, P, np.matrix([obs_point]).T, Q, R)
        result.append(x.getA()[:, 0])
        P_result.append(np.linalg.det(P))
        y_result.append(np.sum(kf.y.getA() ** 2))

        print('x          : {0}'.format(x.getA()[:, 0]))
        if show_working:
            print('x_predicted: {0}'.format(kf.x_predicted[0, 0]))
            print('P_predicted: {0}'.format(kf.P_predicted[0, 0]))

            print('K          : {0}'.format(kf.K[0, 0]))
            print('z          : {0}'.format(kf.z[0, 0]))
            print('y          : {0}'.format(kf.y[0, 0]))

            print('x          : {0}'.format(x[0, 0]))
            print('P          : {0}'.format(P[0, 0]))

    return np.array(result), np.array(P_result), np.array(y_result)


def _demo_multiple_2d_with_inertia(x_init, obs, Q=1e-5, R=0.1 ** 2, show_working=False):
    F = np.matrix([[1., 0., 1., 0.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    H = np.matrix([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]])

    Q = np.matrix([[2., 1., 1., 1.],
                   [1., 2., 1., 1.],
                   [1., 1., 2., 1.],
                   [1., 1., 1., 2.]]) * Q
    R = np.matrix([[2., 1.],
                   [1., 2.]]) * R

    kf = KalmanFilter(F, H)

    x = np.matrix([x_init[0], x_init[1], 0, 0]).T
    P = np.matrix(np.eye(4)) * 10

    result = []
    P_result = []
    y_result = []

    result.append(x.getA()[:, 0])
    P_result.append(np.linalg.det(P))

    for obs_points in obs[1:]:
        # import ipdb; ipdb.set_trace()
        x, P = kf.estimate_multiple(x, P, obs_points, Q, R)
        result.append(x.getA()[:, 0])
        P_result.append(np.linalg.det(P))
        y_result.append(np.sum(kf.y.getA() ** 2))

        print('x          : {0}'.format(x.getA()[:, 0]))
        if show_working:
            print('x_predicted: {0}'.format(kf.x_predicted[0, 0]))
            print('P_predicted: {0}'.format(kf.P_predicted[0, 0]))

            print('K          : {0}'.format(kf.K[0, 0]))
            print('z          : {0}'.format(kf.z[0, 0]))
            print('y          : {0}'.format(kf.y[0, 0]))

            print('x          : {0}'.format(x[0, 0]))
            print('P          : {0}'.format(P[0, 0]))

    return np.array(result), np.array(P_result), np.array(y_result)


def _demo_kalman(x, P, xs, ys, Q, R):
    plt.clf()
    plt.plot(xs, ys, 'ro')
    result = []
    l2k = Linear2DKalman()
    for point in zip(xs, ys):
        x, P = l2k.estimate(x, P, np.matrix(point).T, Q, R)
        result.append(x)
    result = np.array(result)
    kalman_x, kalman_y = result[:, 0, 0], result[:, 1, 0]

    plt.plot(kalman_x, kalman_y, 'g-')
    plt.show()
    return result


def _plot_rts_smoother(rts_smoother, plot_all=True):
    plt.clf()
    filtered_xs = np.array(rts_smoother.filtered_xs)[:, :, 0]
    smoothed_xs = np.array(rts_smoother.xs)[:, :, 0]
    measured_xs = np.array(rts_smoother.zs)

    plt.plot(filtered_xs[:, 0], filtered_xs[:, 1], 'g-', label='filtered')
    plt.plot(smoothed_xs[:, 0], smoothed_xs[:, 1], 'r-', label='smoothed')
    plt.plot(measured_xs[:, 0], measured_xs[:, 1], 'k+', label='measured')

    plt.legend(loc='best')
