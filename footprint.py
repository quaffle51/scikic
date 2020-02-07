import textwrap

from engineering_notation import EngNumber


class Footprint():

    def __init__(self, din, n, w, s, dout, rho, inductance):
        self.din = float(din)
        self.n = int(n)
        self.w = float(w)
        self.s = float(s)
        self.dout = float(dout)
        self.rho = float(rho)
        self.inductance = float(inductance)

        self.via_hole_diam = 0.3  # minimum via hole size is 0.3mm
        self.via_size_x = 0.6  # minimum Via diameter is 0.6mm
        self.via_size_y = 0.6  # minimum Via diameter is 0.6mm

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

        x, y = self.get_xy_positions()
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
            '(fp_text user "din={:.3f} dout={:.3f} n={} w={:.3f} s={:.3f} L={}H rho={:.4f}" \
            (at 0 {})(layer Cmts.User) (effects(font(size 1 1)(thickness 0.15))))\n'.format(
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

    def get_xy_positions(self):
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
