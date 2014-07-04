# Independent.
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, F, H):
        self.F = F
        self.H = H
        self.I = np.matrix(np.eye(F.shape[0])) # identity matrix
        
    def predict(self, x_init, P_init, Q):
        F = self.F

        # Predict
        self.x_predicted = F * x_init
        self.P_predicted = F * P_init * F.T + Q

        return self.x_predicted, self.P_predicted

    def update_multiple(self, x, P, zs, R):
        for z in zs:
            x, P = self.update(x, P, np.matrix([x]).T , R)

        return x, P

    def update(self, x, P, z, R):
        H = self.H
        I = self.I

        # Update
        #import ipdb; ipdb.set_trace()
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

    def estimate_multiple(self, x_init, P_init, zs, Q, R):
        x, P = self.predict(x_init, P_init, Q)
        x, P = self.update_multiple(x, P, zs, R)
        
        return x, P

    def estimate(self, x_init, P_init, z, Q, R):
        x, P = self.predict(x_init, P_init, Q)
        x, P = self.update(x, P, z, R)

        return x, P


class Linear2DKalman(KalmanFilter):
    def __init__(self):
        F = np.matrix([[1., 0., 1., 0.],
                       [0., 1., 0., 1.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])


        H = np.matrix([[1., 0., 0., 0.],
                       [0., 1., 0., 0.]])

        super(Linear2DKalman, self).__init__(F, H)

def demo_voltage(zs=None):
    sz = (50,)
    x_true = -0.37727
    xs = np.ones(sz) * x_true
    if zs == None:
        zs = np.random.normal(0, 0.1, size=sz)
    obs_xs = xs + zs

    result = demo_simple_1d(obs_xs)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, 'k+', label='noisy measurements')
    plt.plot(result, 'b-', label='a posteriori estimate')
    plt.axhline(x_true ,color='g', label='truth value')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')

    return obs_xs, zs, result

def demo_simple_1d(obs_xs, Q=1e-5, R=0.1**2, show_working=False):
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

def demo_2d_with_inertia(noise=0.5, Q=1e-2, R=0.5**2):
    sz = (50,2)
    true_xs = np.linspace(0, sz[0])
    true_xs[25:] = np.ones(25) * 25
    #true_ys = 4 * np.sin(true_xs / 5)
    true_ys = np.zeros_like(true_xs)
    true_ys[25:] = true_xs[:25]

    zs = np.random.normal(0, noise, size=sz)

    obs_xs = true_xs + zs[:, 0]
    #obs_xs = true_xs
    obs_ys = true_ys + zs[:, 1]

    x_result, P_result, y_result = demo_simple_2d_with_inertia(zip(obs_xs, obs_ys), Q, R)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, obs_ys, 'k+', label='noisy measurements')
    plt.plot(x_result[:, 0], x_result[:, 1], 'b-', label='a posteriori estimate')

    plt.figure(2)
    plt.clf()
    plt.plot(y_result, 'b-', label='y')
    #plt.axhline(x_true ,color='g', label='truth value')
    #plt.legend()

    return zip(obs_xs, obs_ys), x_result

def demo_2d(Q=1e-5, R=0.1**2):
    sz = (50,2)
    true_xs = np.linspace(0, sz[0])
    true_ys = true_xs ** 2

    zs = np.random.normal(0, 100, size=sz)

    #obs_xs = true_xs + zs[:, 0]
    obs_xs = true_xs
    obs_ys = true_ys + zs[:, 1]

    Q = np.matrix([[2, 1], [1, 2]]) * Q
    R = np.matrix([[2, 1], [1, 2]]) * R

    result = demo_simple_2d(zip(obs_xs, obs_ys), Q, R)

    plt.figure(1)
    plt.clf()
    plt.plot(obs_xs, obs_ys, 'k+', label='noisy measurements')
    plt.plot(result[:, 0], result[:, 1], 'b-', label='a posteriori estimate')
    #plt.axhline(x_true ,color='g', label='truth value')
    #plt.legend()

    return zip(obs_xs, obs_ys), result

def demo_simple_2d(obs_points, Q=1e-5, R=0.1**2, show_working=False):
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


def demo_simple_2d_with_inertia(x_init, obs_points, Q=1e-5, R=0.1**2, show_working=False):
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
        #import ipdb; ipdb.set_trace()
        x, P = kf.estimate(x, P, np.matrix([obs_point]).T, Q, R)
        result.append(x.getA()[:, 0])
        P_result.append(np.linalg.det(P))
        y_result.append(np.sum(kf.y.getA()**2))

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


def demo_multiple_2d_with_inertia(x_init, obs, Q=1e-5, R=0.1**2, show_working=False):
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
        #import ipdb; ipdb.set_trace()
        x, P = kf.estimate_multiple(x, P, obs_points, Q, R)
        result.append(x.getA()[:, 0])
        P_result.append(np.linalg.det(P))
        y_result.append(np.sum(kf.y.getA()**2))

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


def demo_kalman(x, P, xs, ys, Q, R):
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
