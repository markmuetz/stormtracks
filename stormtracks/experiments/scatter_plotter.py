import numpy as np
import PySide

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

from stormtracks.results import StormtracksResultsManager

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
gx.translate(-10, 0, 0)
w.addItem(gx)
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -10, 0)
w.addItem(gy)
gz = gl.GLGridItem()
gz.translate(0, 0, -10)
w.addItem(gz)

show_changers = False
show_movers = False
show_stormtracks_results = True


if show_stormtracks_results:
    srm = StormtracksResultsManager('plot_results')
    ps1 = srm.get_result(2003, 0, 'scatter_vort_pmin')
    ps2 = srm.get_result(2003, 0, 'scatter_vort_t850')

    plot_attrs = {
        'hu': (1, 0, 0, 1),
        'ts': (0, 0, 1, 1),
        'no': (1, 1, 0, 1),
        'unmatched': (0, 1, 0, 1),
        }

    xmax = -1e99
    xmin = 1e99
    ymax = -1e99
    ymin = 1e99
    zmax = -1e99
    zmin = 1e99

    for key, colour in plot_attrs.items():
        xs = np.array(ps1[key]['xs'])
        ys = np.array(ps1[key]['ys'])
        zs = np.array(ps2[key]['ys'])

        xmax = max(xs.max(), xmax)
        xmin = min(xs.min(), xmin)
        ymax = max(ys.max(), ymax)
        ymin = min(ys.min(), ymin)
        zmax = max(zs.max(), zmax)
        zmin = min(zs.min(), zmin)

    for key, colour in plot_attrs.items():
        xs = np.array(ps1[key]['xs'])
        ys = np.array(ps1[key]['ys'])
        zs = np.array(ps2[key]['ys'])

        xs = 10 * (xs - xmin) / (xmax - xmin)
        ys = 10 * (ys - ymin) / (ymax - ymin)
        zs = 10 * (zs - zmin) / (zmax - zmin)

        positions = np.zeros((len(xs), 3))
        positions[:, 0] = xs
        positions[:, 1] = ys
        positions[:, 2] = zs

        sp2 = gl.GLScatterPlotItem(pos=positions, color=colour, size=5)
        w.addItem(sp2)


phase = 0.
if show_changers:
    pos = np.random.random(size=(100000,3)) - 0.5
    pos *= [20, 20, 20]
    pos[0] = (0,0,0)
    pos = np.array([p for p in pos if (p**2).sum()**0.5 < 10])
    color = np.ones((pos.shape[0], 4))
    d2 = (pos**2).sum(axis=1)**0.5
    size = np.random.random(size=pos.shape[0])*10
    sp2 = gl.GLScatterPlotItem(pos=pos, color=(1,0,1,1), size=size)

    w.addItem(sp2)

if show_movers:
    n = 300
    pos3 = np.zeros((n,n,3))
    pos3[:,:,:2] = np.mgrid[:n, :n].transpose(1,2,0) * [-0.1,0.1]
    pos3 = pos3.reshape(n**2,3)
    d3 = (pos3**2).sum(axis=1)**0.5

    sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

    w.addItem(sp3)


def update():
    return
    global phase, sp2, d2
    if show_changers:
        ## update volume colors
        s = -np.cos(d2 * 0.4 + phase)
        color = np.empty((len(d2),4), dtype=np.float32)
        color[:,3] = np.clip(s * 0.3, 0, 1)
        color[:,0] = np.clip((1 - s) * 1.0, 0, 1)
        color[:,1] = np.clip(s * 0.0, 0, 1)
        color[:,2] = np.clip(s * 1.0, 0, 1)
        sp2.setData(color=color)

    global sp3, d3, pos3
    if show_movers:
        ## update surface positions and colors
        z = -np.cos(d3*2+phase)
        pos3[:,2] = z
        color = np.empty((len(d3),4), dtype=np.float32)
        color[:,3] = 1.
        color[:,0] = 0.
        color[:,1] = 1.
        color[:,2] = 1.
        # color[:,0] = np.clip(z * 3.0, 0, 1)
        # color[:,1] = np.clip(z * 1.0, 0, 1)
        # color[:,2] = np.clip(z * 3, 0, 1)
        sp3.setData(pos=pos3, color=color)

    # global gx
    # gx.rotate(0.7, 0, 1, 0)
    phase -= 0.1

    
    
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


