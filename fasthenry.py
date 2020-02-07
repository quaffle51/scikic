import math
import os
import subprocess
import sys
from subprocess import Popen

import numpy as np

import par

useNLopt = True
try:  # check if nlopt is available
    import nlopt
except ImportError:
    print("nlopt not unavailable!")
    useNLopt = False


class Fasthenry():

    def __init__(self, din, n, w, s, h):
        self.din = din
        self.n = n
        self.w = w
        self.s = s
        self.h = h
        self.rho = 0
        self.dout = 0
        self.v = True

    def determine_impedance_of_square_planar_inductor_using_fasthenry(self):
        inductance = -1
        message = 'Ready'
        nodes = self.get_start_end_coords()
        x, y = self.nodes_2_x_and_y(nodes)
        self.generate_fasthenry_input_file(x, y)
        inp = "planar_inductor.inp"
        if os.path.exists(inp):
            logfile = 'logfile.txt'
            output_to_logfile = open(logfile, 'w+')
            if os.path.exists('Zc.mat'):
                os.remove('Zc.mat')
            p = Popen(["fasthenry", inp], stdout=output_to_logfile, stderr=subprocess.PIPE, universal_newlines=True)
            p.wait()

            if os.path.exists('Zc.mat'):
                lines = []
                f = open('Zc.mat', 'r')
                for line in f:
                    lines.append(line)
                f.close()
                z = lines[-1].split()
                z = complex(float(z[0]), float(z[1][:-1]))
                inductance = z.imag / (2 * math.pi * 200000)
            else:
                message = 'Zc.mat NOT generated'
        else:
            message = "The file 'planar_inductor.inp' the input file to Fasthenry doesn't exist!"

        return inductance, message

    def get_start_end_coords(self):
        din = self.din
        w = self.w
        s = self.s
        n = int(self.n)
        x, y = self.get_x_y_positions()
        x0y0 = []
        for i in range(len(x)):
            x0y0.append(par.Point(x[i], y[i]))
            x0y0[i].slide_xy(-(w + s) / 4.0, 0.0)
        base_width = din + w - s
        widths = []
        for i in range(2 * n):
            widths.append(base_width + i * (w + s))
        nodes = []
        for i in range(0, len(x0y0), 2):
            p = x0y0[i].clone()
            p.slide_xy(-widths[int(i / 2)] / 2 + w / 2, 0)
            q = x0y0[i].clone()
            q.slide_xy(widths[int(i / 2)] / 2 - w / 2, 0)
            nodes.append(p)
            nodes.append(q)
        for i in range(2, len(nodes), 4):
            tmp = nodes[i]
            nodes[i] = nodes[i + 1]
            nodes[i + 1] = tmp

        return nodes

    def get_x_y_positions(self):

        din = self.din
        w = self.w
        s = self.s
        n = int(self.n)

        x = [0 for _ in range(4 * n)]

        for i in range(len(x)):
            dv, md = divmod(i, 4)
            if md == 3:
                x[i] = -(din / 2 + dv * (w + s) + w / 2)
                x[i - 2] = - x[i]
            if md == 0:
                x[i] = (w + s) / 2
        y = x[1:] + [0]

        return x, y

    def generate_fasthenry_input_file(self, start_x, start_y):

        n = int(self.n) * 4 + 1
        w = self.w
        h = self.h
        nwinc = 1
        nhinc = 1

        fasthenry = ''
        title = '** Planar Inductor **\n'
        fasthenry += title
        units = '* The following line names millimeters as the length units for the rest\n'
        units += '* of the file.\n'
        units += '.Units MM\n\n'
        fasthenry += units
        defaults = ''
        defaults += '* Make z=0 the default z coordinate and copper the default conductivity.\n'
        defaults += '* Note that the conductivity is in units of 1/(mm*Ohms), and not 1/(m*Ohms)\n'
        defaults += '* since the default units are millimetres\n'
        defaults += '.Default z=0 sigma=5.9595e4\n\n'
        fasthenry += defaults

        nodes = ''
        nodes += '* The nodes of the planar inductor (z=0 is the default)\n'
        node_coordinates_x = []
        node_coordinates_y = []
        for i in range(len(start_x)):
            node_coordinates_x.append('{}'.format(start_x[i]))
            node_coordinates_y.append('{}'.format(start_y[i]))
        node_coordinates_x.append('{}'.format(start_x[-1]))
        node_coordinates_y.append('{}'.format(-1 * start_y[-1]))

        for node in range(len(node_coordinates_x)):
            nodes += '{:10} x={:10} y={:10}\n'.format('N' + str(node + 1), node_coordinates_x[node],
                                                      node_coordinates_y[node])

        nodes += '\n\n* The segments connecting the nodes\n'
        for node in range(n - 1):
            nodes += 'E{0} N{0} N{1} w={2} h={3} nhinc={4} nwinc={5}\n'.format(node + 1, node + 2, w, h, nhinc, nwinc)
        nodes += '\n*  Define one input \'port\' of the network\n'
        nodes += '.external N{} N{}\n'.format(1, n)
        nodes += '\n* Frequency range of interest\n'
        nodes += '.freq fmin=200000 fmax=200000 ndec=1\n\n'
        nodes += '.end\n'

        fasthenry += nodes

        filename = "planar_inductor.inp"
        f = open(filename, "w+")
        f.write(fasthenry)
        f.close()

        return fasthenry

    def nodes_2_x_and_y(self, nodes):
        x = []
        y = []
        for node in nodes:
            x.append(np.round(node.x, 3))
            y.append(np.round(node.y, 3))
        return x, y

    def runOptimization(self, gui, root):
        target_ind = float(gui.entry_t_ind.get())
        target_rho = float(gui.entry_t_rho.get())
        n_min = float(gui.min_n.get())
        w_min = float(gui.min_w.get())
        s_min = float(gui.min_s.get())
        din_min = float(gui.min_din.get())

        def errfunc(x, grad):
            if grad.size > 0:
                grad = None
            gui.s.set(x[0])
            self.s = x[0]
            gui.w.set(x[1])
            self.w = x[1]
            gui.din.set(x[2])
            self.din = x[2]
            gui.n.set(int(x[3]))
            self.n = int(x[3])
            gui.dout.set(float(gui.din.get()) + 2 * float(gui.n.get()) * float(gui.w.get()) + 2 * (
                    float(gui.n.get()) - 1) * float(float(gui.s.get())))
            gui.rho.set((float(gui.dout.get()) - float(gui.din.get())) / (float(gui.dout.get()) + float(gui.din.get())))
            root.update()

            ind, message = self.determine_impedance_of_square_planar_inductor_using_fasthenry()
            self.L = ind
            err = math.fabs(ind - target_ind)
            return err

        def din_constraint(x, grad):
            if grad.size > 0:
                grad = None
            s = x[0]
            din = x[2]
            return 2 * s + 0.6 - din

        def rho_constraint(x, grad):
            if grad.size > 0:
                grad = None
            s = x[0]
            w = x[1]
            din = x[2]
            n = int(x[3])
            dout = din + 2 * n * w + 2 * (n - 1) * s
            rho = (dout - din) / (dout + din)
            return target_rho - rho

        opt = nlopt.opt(nlopt.LN_COBYLA, 4)  # opt = nlopt.opt(algorithm, n), (n, the number of optimization parameters)

        opt.set_min_objective(errfunc)
        opt.add_inequality_constraint(lambda x, grad: din_constraint(x, grad), 1e-8)
        opt.add_inequality_constraint(lambda x, grad: rho_constraint(x, grad), 1e-8)
        opt.set_lower_bounds([s_min, w_min, din_min, n_min])
        opt.set_upper_bounds([1, 1, 40, 100])
        opt.set_xtol_rel(1e-6)

        n = n_min
        din = din_min
        w = w_min
        s = s_min

        x = opt.optimize([s, w, din, n])

        minf = opt.last_optimum_value()
        return x[0]


def main(args):
    din = 0.865
    n = 1
    s = 0.128
    w = 0.128
    h = 0.035
    fh = Fasthenry(din, n, w, s, h)
    inductance, message = fh.determine_impedance_of_square_planar_inductor_using_fasthenry()
    print('Inductance = {}'.format(inductance))
    print('Message = {}'.format(message))
    fh.runOptimization()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
