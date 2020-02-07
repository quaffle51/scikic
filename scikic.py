#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  scikic.py
#
#  Copyright 2020 Gordon <gordon@gordon-MS-7866>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


class Scikic:

    def __init__(self):
        # Single layer Planar spiral coil inductor calculator
        # See: the square planar inductor at
        # http://www.circuits.dk/calculator_planar_coil_inductor.htm
        #
        # Note: units of length are millimetres (mm)
        #
        # n, number of turns or windings, N>=2
        # dint, inner dimension
        # s, gap or spacing between windings
        # w, conductor width
        # h, conductor height or thickness, hidden in Figure 1 of the paper.

        # dimension associated with the via: Size X: 0.6, Size y: 0.6; Hole shape = circular, Hole Size: 0.3
        # See: https://jlcpcb.com/capabilities/Capabilities
        self.via_hole_diam = 0.3  # minimum via hole size is 0.3mm
        self.via_size_x = 0.6  # minimum Via diameter is 0.6mm
        self.via_size_y = 0.6  # minimum Via diameter is 0.6mm
        # Fasthenry parameters
        self.nwinc: int = 1  # see Fasthenry manual
        self.nhinc: int = 1

        self.parseCommandLineArgs()

    def getXYPositions(self):
        # din = inner length
        # w = track width
        # s = spacing between tracks
        # number of turns

        din = self.din
        w = self.w
        s = self.s
        n = self.n

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

    def getLengths(self):

        din = self.din
        w = self.w
        s = self.s
        n = self.n
        l = [0 for _ in range(int((4 * n) / 2))]
        count = 1
        for i in range(len(l)):
            l[i] = din + count * w + (count - 2) * (s)
            count += 1

        return l

    def getStartEndCoords(self):

        din = self.din
        w = self.w
        s = self.s
        n = self.n
        x, y = self.getXYPositions()
        x0y0 = []
        for i in range(len(x)):
            x0y0.append(par.Point(x[i], y[i]))
            x0y0[i].slide_xy(-(w + s) / 4.0, 0.0)
        nodes = []
        base_width = din + w - s
        widths = []
        nodes = []
        for i in range(2 * n):
            widths.append(base_width + i * (w + s))
        nodes = []
        for i in range(0, len(x0y0), 2):
            # print('width of side {:>2} is {:.3f} @ ({:.3f},{:.3f})]'.format(i+1,widths[int(i/2)], x0y0[i].x, x0y0[i].y))
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

    def printNodes(self, nodes):
        print('nodes:')
        for i in range(len(nodes)):
            print('n{:>2}=({:>6.3f},{:>6.3f})'.format(i, nodes[i].x, nodes[i].y))

    def getEndCoords(self, length, x0):

        w = self.w
        return (x0 - length / 2 + w / 2), (x0 + length / 2 - w / 2)

    def getWidths(self, lengths, trackWidth):
        widths = []

        for i in range(len(lengths)):
            widths.append(lengths[i])
            widths.append(trackWidth)

        return widths

    def getHeights(self, widths):
        return widths[1:] + [widths[-2]]

    def formKiCadFootprint(self):

        din = self.din
        trackWidth = self.w
        s = self.s
        n = self.n
        via_size = self.via_size_x

        x, y = self.getXYPositions()
        l = self.getLengths()
        widths = self.getWidths(l, trackWidth)
        heights = self.getHeights(widths)
        footprint = textwrap.dedent("""\
			(module mohan (layer F.Cu) (tedit 5E1D9573)
			  (fp_text reference REF** (at 0 0) (layer F.SilkS)
				(effects (font (size 1 1) (thickness 0.15)))
			  )
			  (fp_text value mohan (at 0 0) (layer F.Fab)
				(effects (font (size 1 1) (thickness 0.15)))
			  )			
		""")

        line_txt = '  (pad {} smd rect (at {:.3f} {:.3f}) (size {:.3f} {:.3f}) (layers F.Cu F.Paste F.Mask))\n'

        for i in range(len(x)):
            footprint += line_txt.format(i + 3, x[i], y[i], widths[i], heights[i])

        l1 = din - s + trackWidth
        p = x[0] - l1 / 2
        q = -(din / 2 + n * trackWidth + (n - 1) * s) + trackWidth / 2
        pad = '  (pad {} thru_hole rect (at {:.3f} {:.3f})(size {} {}) (drill 0.3)(layers *.Cu))\n'
        footprint += pad.format(1, p + via_size / 2, y[0] - via_size / 2, via_size, via_size)
        footprint += pad.format(2, q - via_size / 2 + trackWidth / 2, -q + trackWidth / 2 + via_size / 4,
                                via_size, via_size)
        line_txt = '  (fp_line (start {:.3f} {:.3f}) (end {:.3f} {:.3f}) (layer F.CrtYd) (width 0.12))\n'
        q = (2 * n * trackWidth + 2 * (n - 1) * s + din) / 2 + s + via_size  # basis for corner coordinates
        footprint += line_txt.format(-q, -q, q, -q)
        footprint += line_txt.format(q, -q, q, q)
        footprint += line_txt.format(q, q, -q, q)
        footprint += line_txt.format(-q, q, -q, -q)
        line_txt = '  (fp_line (start {:.3f} {:.3f}) (end {:.3f} {:.3f}) (layer F.SilkS) (width 0.12))\n'
        footprint += line_txt.format(-q, -q, q, -q)
        footprint += line_txt.format(q, -q, q, q)
        footprint += line_txt.format(q, q, -q, q)
        footprint += line_txt.format(-q, q, -q, -q)
        footprint += \
            '(fp_text user "din={:.3f} dout={:.3f} n={} w={:.3f} s={:.3f} L={}H rho={:.3f}" (at 0 {})(layer Cmts.User) (effects(font(size 1 1)(thickness 0.15))))\n'.format(
                self.din,
                self.dout,
                self.n,
                self.w,
                self.s,
                EngNumber(self.inductance, precision=3),
                self.rho,
                q + 1)
        footprint += ")\n"

        return footprint

    def calculateInductance(self):
        a = [1.62e-3, -1.21, -0.147, 2.40, 1.78, -0.030]  # values from Table III of paper.
        b = a[0]

        din = self.din * 1000
        dout = self.dout * 1000
        davg = np.divide((dout + din), 2)
        w = self.w * 1000
        n = self.n
        s = self.s * 1000
        # Monomial Expression
        lmon = np.multiply(b, np.power(dout, a[1]))
        lmon = np.multiply(lmon, np.power(w, a[2]))
        lmon = np.multiply(lmon, np.power(davg, a[3]))
        lmon = np.multiply(lmon, np.power(n, a[4]))
        lmon = np.multiply(lmon, np.power(s, a[5]))

        # Modified Wheeler
        k1 = 2.34
        k2 = 2.75
        rho = self.rho
        u = 4e-7 * np.pi
        lmw = np.multiply(u, np.power(n, 2))
        lmw = np.multiply(lmw, k1)
        lmw = np.multiply(lmw, davg)
        lmw = np.divide(lmw, (1 + np.multiply(k2, rho)))
        lmw = np.multiply(lmw, 1e3)

        #  current sheet
        c1 = 1.27
        c2 = 2.07
        c3 = 0.18
        c4 = 0.13
        lgmd = np.multiply(u, np.power(n, 2))  # u*n^2
        lgmd = np.multiply(lgmd, davg)  # u*n^2*davg
        lgmd = np.multiply(lgmd, c1)  # u*n^2*davg*c1
        lgmd = np.divide(lgmd, 2)  # u*n^2*davg*c1/2
        lgmd = np.multiply(lgmd, np.log(np.divide(c2, rho)) + np.multiply(c3, rho) + np.multiply(c4, np.power(rho, 2)))
        lgmd = np.multiply(lgmd, 1e3)

        return np.multiply(lmw, 1e-09), np.multiply(lgmd, 1e-09), np.multiply(lmon, 1e-09)

    def parseCommandLineArgs(self):

        tag = textwrap.dedent( \
            """
            {}{}
            """).format("\\\\", "\\")

        examples = textwrap.dedent( \
            """
              examples:
                %(prog)s -N {0:>2} -Ai {4} -s {8} -g {9} -t {10}
                %(prog)s -N {1:>2} -Ai {5} -s {8} -g {9} -t {10}
                %(prog)s -N {2:>2} -Ai {6} -s {8} -g {9} -t {10}
                %(prog)s -N {3:>2} -Ai {7} -s {8} -g {9} -t {10}
            """.format(2, 5, 10, 15, 47.2, 38.8, 24.8, 10.8, 0.7, 0.7, 0.035))

        parser = argparse.ArgumentParser(description='KiCad PCB Footprint Utility.', \
                                         epilog=examples, \
                                         formatter_class=RawTextHelpFormatter)
        group_2 = parser.add_argument_group('Optimization')
        group_1 = parser.add_argument_group('Coil parameters')

        group_1.add_argument('-n', '--number_of_turns', help=u'number of turns or windings, n \u2265 1', type=int,
                             default=1, action='store', dest='n'),
        group_1.add_argument('-din', '--inside_dimension', help='internal dimension, mm', default=0.856, type=float,
                             action='store',
                             dest='din')
        group_1.add_argument('-w', '--trace_width', help='conductor width, mm', default=0.128, type=float,
                             action='store', dest='w')
        group_1.add_argument('-s', '--trace_spacing', help='spacing between windings, mm', default=0.128, type=float,
                             action='store',
                             dest='s')
        group_1.add_argument('-t', '--trace_height', help='trace thickness or height, mm', default=0.035, type=float,
                             action='store',
                             dest='t')
        group_1.add_argument('-v', '--verbosity', dest='v', help='output additional results', action='store_true')
        group_1.add_argument('-f', '--footprint', dest='f', help='save kicad pcb footprint', action='store_true')
        group_2.add_argument('-o', '--optimize', help='optimize coil parameters to give the required inductance',
                             default=0, type=float, action='store', dest='o')
        args = vars(parser.parse_args())

        self.n = args['n']
        self.din = args['din']
        self.s = args['s']
        self.w = args['w']
        self.h = args['t']
        self.o = args['o']
        self.f = args['f']
        self.v = args['v']

        if self.n < 1:
            print('The number of turns has to be greater than or equal to 1.')
            sys.exit(-1)

        if self.w < 0.128:
            print('The track width has to be greater than or equal to 0.128mm or 5mil.')
            sys.exit(-1)

        if self.s < 0.128:
            print('The track spacing has to be greater than or equal to 0.128mm or 5mil.')
            sys.exit(-1)

        if self.din < 2 * self.s + self.via_size_x:
            print(
                'The inner dimension of the coil must be greater than {:.3f} mm.'.format(2 * self.s + self.via_size_x))
            sys.exit(-1)

        self.N = 4 * self.n + 1  # number of sides
        self.dout = self.din + 2 * (self.n * self.w + (self.n - 1) * self.s)
        self.rho = (self.dout - self.din) / (self.dout + self.din)
        if self.v:
            print(tag)
            print('dout = {}'.format(EngNumber(self.dout, precision=3)))
            print('n={}'.format(args['n']))
            print('din={}'.format(args['din']))
            print('s={}'.format(args['s']))
            print('w={}'.format(args['w']))
            print('t={}'.format(args['t']))
            print('rho={:.4f}'.format(self.rho))

    def generateFasthenryInputFile(self, start_x, start_y):

        n = self.n * 4 + 1
        w = self.w
        h = self.h
        nwinc = self.nwinc
        nhinc = self.nhinc

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
        nodeCoordinates_x = []
        nodeCoordinates_y = []
        for i in range(len(start_x)):
            nodeCoordinates_x.append('{}'.format(start_x[i]))
            nodeCoordinates_y.append('{}'.format(start_y[i]))
        nodeCoordinates_x.append('{}'.format(start_x[-1]))
        nodeCoordinates_y.append('{}'.format(-1 * start_y[-1]))

        for node in range(len(nodeCoordinates_x)):
            nodes += '{:10} x={:10} y={:10}\n'.format('N' + str(node + 1), nodeCoordinates_x[node],
                                                      nodeCoordinates_y[node])

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

    def nodes2XAndY(self, nodes):
        x = []
        y = []
        for node in nodes:
            x.append(np.round(node.x, 3))
            y.append(np.round(node.y, 3))
        return x, y

    def determineImpedanceOfSquarePlanarInductorUsingFasthenry(self):
        nodes = self.getStartEndCoords()
        x, y = self.nodes2XAndY(nodes)
        self.generateFasthenryInputFile(x, y)
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
                self.inductance = z.imag / (2 * math.pi * 200000)
                if self.v:
                    print('Zc.mat successfully created.')
                    print('real part of impedance = {}'.format(z.real))
                    print('imag part of impedance = {}'.format(z.imag))
                    print('Fasthenry determined Inductance of the planar inductor to be: {}H'.format(
                        EngNumber(self.inductance, precision=3)))
                else:
                    print(self.inductance)
            else:
                print('Zc.mat NOT generated!')
                sys.exit()
        else:
            print("The file 'planar_inductor.inp' the input file to Fasthenry doesn't exist!")
            sys.exit()

        return self.inductance

    def runOptimization(self):

        n = 1
        din = 0.856
        w = 0.128
        s = 0.128
        targetInd: float = self.o

        def errfunc(x, grad):
            if grad.size > 0:
                grad = None
            self.s   = x[0]
            self.w   = x[1]
            self.din = x[2]
            self.n   = int(x[3])
            self.dout = self.din + 2 * self.n * self.w + 2 * (self.n - 1) * self.s
            self.rho  = (self.dout - self.din)/(self.dout + self.din)
            print('\n=====> s={}'.format(EngNumber(x[0], precision=3)))
            print('=====> w={}'.format(EngNumber(x[1], precision=3)))
            print('=====> din={}'.format(EngNumber(x[2], precision=3)))
            print('=====> n={}'.format(EngNumber(x[3], precision=3)))
            print('=====> dout={}'.format(self.dout))
            print('=====> rho={:.4f}'.format(self.rho))

            ind = self.determineImpedanceOfSquarePlanarInductorUsingFasthenry()
            print('=====> ind={}'.format(ind))
            self.L = ind
            err = math.fabs(ind - targetInd)
            return err

        def din_constraint(x,grad):
            if grad.size > 0:
                grad = None
            s = x[0]
            din = x[2]
            return 2*s + 0.6 - din

        def rho_constraint(x,grad):
            if grad.size > 0:
                grad = None
            s   = x[0]
            w   = x[1]
            din = x[2]
            n   = int(x[3])
            dout = din + 2 * n * w + 2 * (n - 1) * s
            rho  = (dout - din)/(dout + din)
            return 0.9 - rho


        opt = nlopt.opt(nlopt.LN_COBYLA, 4)  # opt = nlopt.opt(algorithm, n), (n, the number of optimization parameters)
        s_min = 0.128
        w_min = 0.128
        din_min = 0.856
        n_min = 1

        opt.set_min_objective(errfunc)
        opt.add_inequality_constraint(lambda x, grad: din_constraint(x, grad), 1e-8)
        opt.add_inequality_constraint(lambda x, grad: rho_constraint(x, grad), 1e-8)
        opt.set_lower_bounds([s_min, w_min, din_min, n_min])
        opt.set_upper_bounds([    1,     1,      40,   100])
        opt.set_xtol_rel(1e-6)
        x = opt.optimize([s, w, din, n])
        
        minf = opt.last_optimum_value()
        print("optimum at ", x[0])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())
        print('\nOptimized values for an inductor with impedance {}H are:'.format(EngNumber(self.L, precision=3)))
        print('** s={}'.format(EngNumber(self.s, precision=0)))
        print('** w={}'.format(EngNumber(self.w, precision=0)))
        print('** din={}'.format(EngNumber(self.din, precision=3)))
        print('** n={}'.format(EngNumber(self.n, precision=0)))
        print('** dout={}'.format(EngNumber(self.din + 2 * self.n * self.w + 2 * (self.n - 1) * self.s, precision=3)))
        print('** rho={:.4f}'.format(self.rho))
        self.formKiCadFootprint()
        return x[0]


def main(args):
    sk = Scikic()
    lmw, lgmd, lmon = sk.calculateInductance()
    if sk.v:
        print(
            'lmw = {}H, lgmd={}H. lmon={}H'.format(EngNumber(lmw, precision=3), EngNumber(lgmd, precision=3),
                                                   EngNumber(lmon, precision=3)))
    sk.determineImpedanceOfSquarePlanarInductorUsingFasthenry()
    footprint = sk.formKiCadFootprint()
    if sk.o:
        sk.runOptimization()
    if sk.f:
        filename = 'planar_inductor_{:.3f}_{:.3f}_{:.3f}_{}.kicad_mod'.format(sk.din, sk.w, sk.s, sk.n)
        f = open(filename, 'w+')
        f.write(footprint)
        f.close()
        if sk.v:
            print('Footprint saved to file {}'.format(filename))
    return 0


if __name__ == '__main__':
    import sys
    import textwrap
    from engineering_notation import EngNumber
    import argparse
    from argparse import RawTextHelpFormatter
    import os
    import math
    import subprocess
    from subprocess import Popen, PIPE
    import par
    import numpy as np
    import scipy.optimize as optimize
    import minizinc

    useNLopt = True
    try:  # check if nlopt is available
        import nlopt
    except ImportError:
        print("nlopt not unavailable!")
        useNLopt = False

    sys.exit(main(sys.argv))
