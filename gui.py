import math
from tkinter import *
from tkinter import ttk

import numpy as np

import fasthenry as fh
import footprint as fp
import tooltip   as tp


class Gui(ttk.Frame):

    def __init__(self, master):
        self.validate = (master.register(self.validate_entry), '%d', '%S', '%W')
        self.r = re.compile('^[-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?$')  # matches a floating point number
        ttk.Frame.__init__(self, master)

        self.padding = 10
        self.master = master
        # default values
        self.dft_n = StringVar()
        self.dft_din = StringVar()
        self.dft_w = StringVar()
        self.dft_s = StringVar()

        self.dft_n.set(41)
        self.dft_din.set(0.8621)
        self.dft_w.set(0.1315)
        self.dft_s.set(0.1280)

        self.min_n = StringVar()
        self.min_w = StringVar()
        self.min_s = StringVar()
        self.min_din = StringVar()
        self.min_c = StringVar()

        self.min_n.set(1)
        self.min_w.set(0.128)
        self.min_s.set(0.128)
        self.min_din.set(2 * float(self.min_s.get()) + 0.6)
        self.min_c.set(0.035)

        self.mins = [self.min_n, self.min_din, self.min_w, self.min_s]

        self.dfts = [self.dft_n.get(), self.dft_din.get(), self.dft_w.get(), self.dft_s.get()]

        self.n = StringVar()
        self.w = StringVar()
        self.s = StringVar()
        self.din = StringVar()
        self.c = StringVar()

        self.rho = StringVar()
        self.dout = StringVar()

        self.target_ind = StringVar()
        self.target_ind.set('')

        self.target_rho = StringVar()
        self.target_rho.set(0.8)

        master.title("Square Planar Inductor KiCad Footprint Generator")

        # Input parameter frame
        frame_1 = ttk.LabelFrame(master, text=' Inductance from user defined parameters: ')
        frame_1.grid(row=0, column=0, stick='wn', padx=self.padding, pady=self.padding)
        self.input_frame(frame_1)

        # Minimum values frame
        frame_2 = ttk.LabelFrame(master, text=' PCB capability: ')
        frame_2.grid(row=0, column=1, stick='wn', padx=self.padding, pady=self.padding)
        self.minimum_params_frame(frame_2)

        # Fasthenry solver frame
        frame_3 = ttk.LabelFrame(master, text=' Parameters from user defined inductance ')
        frame_3.grid(row=1, column=1, sticky='wn', padx=self.padding, pady=self.padding)
        self.solver_frame(frame_3)

        # Results frame
        frame_4 = ttk.LabelFrame(master, text=' Results of calculations ')
        frame_4.grid(row=1, column=0, stick='wn', padx=self.padding, pady=self.padding)
        self.calculation_frame(frame_4)

        # Information frame
        self.message = StringVar()
        self.default_message = 'Ready'
        self.message.set(self.default_message)
        frame_4 = ttk.Frame(master, borderwidth=10)
        frame_4.grid(row=3, column=0, padx=self.padding, pady=self.padding)
        self.message_frame(frame_4)

    def validate_entry(self, _d_, _S_, _W_):
        bint = _S_.isdigit() and (_W_ == str(self.entry_n_min) or _W_ == str(self.entry_n))
        bflt = _S_ in '+-.eE0123456789' and not (_W_ == str(self.entry_n_min) or _W_ == str(self.entry_n))
        if int(_d_) == 1:
            if bint:
                return True
            elif bflt:
                return True
            else:
                return False
        else:
            return True

    def validate_args(self):
        at_least_one_is_empty = \
            len(self.entry_n.get()) == 0 or \
            len(self.entry_d.get()) == 0 or \
            len(self.entry_w.get()) == 0 or \
            len(self.entry_s.get()) == 0 or \
            len(self.entry_c.get()) == 0 or \
            len(self.entry_n_min.get()) == 0 or \
            len(self.entry_d_min.get()) == 0 or \
            len(self.entry_w_min.get()) == 0 or \
            len(self.entry_s_min.get()) == 0

        if not at_least_one_is_empty:
            are_floats = \
                self.r.match(self.entry_d.get()) and \
                self.r.match(self.entry_w.get()) and \
                self.r.match(self.entry_s.get()) and \
                self.r.match(self.entry_c.get()) and \
                self.r.match(self.entry_d_min.get()) and \
                self.r.match(self.entry_w_min.get()) and \
                self.r.match(self.entry_s_min.get())
        else:
            self.message.set('Please enter values ...')
            are_floats = False

        if are_floats:
            test = True
            # test = int(self.entry_n.get()) >= int(self.entry_n_min.get())
            # test = test and int(self.entry_n_min.get()) > 0
            test = test and float(self.entry_d.get()) >= float(self.entry_d_min.get())
            test = test and float(self.entry_d_min.get()) > 0
            test = test and float(self.entry_w.get()) >= float(self.entry_w_min.get())
            test = test and float(self.entry_w_min.get()) > 0
            test = test and float(self.entry_s.get()) >= float(self.entry_s_min.get())
            test = test and float(self.entry_s_min.get()) > 0
            test = test and float(self.entry_c.get()) > 0.0
        else:
            self.message.set('An entered value is not a valid float')
            test = False

        if not at_least_one_is_empty and are_floats and test:
            return True
        else:
            self.message.set('Error: please check the entered values')
            return False

    def solver_frame(self, frame):
        self.label_t_ind = ttk.Label(frame, text='Required inductance:')
        self.label_t_ind.grid(row=2, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_t_ind = ttk.Entry(frame, width=20, textvariable=self.target_ind, validate='all',
                                     validatecommand=self.validate)
        self.entry_t_ind.grid(row=2, column=3)
        self.label_t_ind_units = ttk.Label(frame, text="H")
        self.label_t_ind_units.grid(row=2, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_t_ind, text=u'The required inductance in H', wraplength=200)

        self.label_t_rho = ttk.Label(frame, text='Required rho:')
        self.label_t_rho.grid(row=3, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_t_rho = ttk.Entry(frame, width=20, textvariable=self.target_rho, validate='all',
                                     validatecommand=self.validate)
        self.entry_t_rho.grid(row=3, column=3)
        self.label_t_rho_units = ttk.Label(frame, text="H")
        self.label_t_rho_units.grid(row=3, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_t_rho, text=u'Fill factor, \u03C1=(dout-din)/(dout+din)', wraplength=200)

        self.button_fh = ttk.Button(frame, text='Find Parameters', state=NORMAL)
        self.button_fh.grid(row=4, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.button_fh.config(command=self.callback_solver)
        tp.Tooltip(self.button_fh,
                   text='Determine the parameters for the given inductance. Experiment with the value of "rho" to try and reduce the outer dimension',
                   wraplength=100)

    def minimum_params_frame(self, frame):

        self.label_n_min = ttk.Label(frame, text='Minimum number of turns:')
        self.label_n_min.grid(row=0, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_n_min = ttk.Entry(frame, width=20, textvariable=self.min_n, validate='all',
                                     validatecommand=self.validate)
        self.entry_n_min.grid(row=0, column=3)
        self.label_n_min_units = ttk.Label(frame, text="")
        self.label_n_min_units.grid(row=0, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_n_min, text=u'The minimum number of turns allowed, an integer.', wraplength=200)

        self.label_d_min = ttk.Label(frame, text='Minimum inner dimension:')
        self.label_d_min.grid(row=2, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_d_min = ttk.Entry(frame, width=20, textvariable=self.min_din, validate='all',
                                     validatecommand=self.validate)
        self.entry_d_min.grid(row=2, column=3)
        self.label_d_min_units = ttk.Label(frame, text="mm")
        self.label_d_min_units.grid(row=2, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_d_min, text=u'The minimum inner dimension based on via dimensions. You will not noramally need to change this value.', wraplength=200)

        self.label_w_min = ttk.Label(frame, text='Minimum track width:')
        self.label_w_min.grid(row=4, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_w_min = ttk.Entry(frame, width=20, textvariable=self.min_w, validate='all',
                                     validatecommand=self.validate)
        self.entry_w_min.grid(row=4, column=3)
        self.label_w_min_units = ttk.Label(frame, text="mm")
        self.label_w_min_units.grid(row=4, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_w_min, text=u'The minimum track width. This value is obtained from the PCB manufacturer.', wraplength=200)

        self.label_s_min = ttk.Label(frame, text='Minimum spacing between tracks:')
        self.label_s_min.grid(row=6, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_s_min = ttk.Entry(frame, width=20, textvariable=self.min_s, validate='all',
                                     validatecommand=self.validate)
        self.entry_s_min.grid(row=6, column=3)
        self.label_s_min_units = ttk.Label(frame, text="mm")
        self.label_s_min_units.grid(row=6, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_s_min, text=u'The minimum spacing between turns. This value is obtained from the PCB manufacturer.', wraplength=200)

        self.label_c = ttk.Label(frame, text='Track thickness:')
        self.label_c.grid(row=8, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_c = ttk.Entry(frame, width=20, textvariable=self.min_c, validate='all',
                                 validatecommand=self.validate)
        self.entry_c.grid(row=8, column=3)
        self.label_c_units = ttk.Label(frame, text="mm")
        self.label_c_units.grid(row=8, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_c, text=u'The track thickness for the type of PCB substrate being used. It is only used by FastHenry in determining the parameters for a given inductance.', wraplength=200)

    def input_frame(self, frame):
        self.label_n = ttk.Label(frame, text="Number of turns (n):", background='')
        self.label_n.grid(row=1, column=1, padx=self.padding, pady=self.padding, sticky='wn')
        self.label_d = ttk.Label(frame, text="Inner dimension (din)")
        self.label_d.grid(row=3, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.label_w = ttk.Label(frame, text="Track width (w:)")
        self.label_w.grid(row=5, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.label_s = ttk.Label(frame, text="Spacing between turns (s):")
        self.label_s.grid(row=7, column=1, padx=self.padding, pady=self.padding, sticky='w')

        self.entry_n = ttk.Entry(frame, width=20, textvariable=self.n, validate='all',
                                 validatecommand=self.validate)
        self.entry_n.grid(row=1, column=3)
        self.entry_n.focus()
        self.entry_d = ttk.Entry(frame, width=20, textvariable=self.din, validate='all',
                                 validatecommand=self.validate)
        self.entry_d.grid(row=3, column=3)
        self.entry_w = ttk.Entry(frame, width=20, textvariable=self.w, validate='all',
                                 validatecommand=self.validate)
        self.entry_w.grid(row=5, column=3)
        self.entry_s = ttk.Entry(frame, width=20, textvariable=self.s, validate='all',
                                 validatecommand=self.validate)
        self.entry_s.grid(row=7, column=3)

        self.label_n_units = ttk.Label(frame, text="")
        self.label_n_units.grid(row=1, column=5, padx=self.padding, pady=self.padding, sticky='e')
        self.label_d_units = ttk.Label(frame, text="mm")
        self.label_d_units.grid(row=3, column=5, padx=self.padding, pady=self.padding, sticky='e')
        self.label_w_units = ttk.Label(frame, text="mm")
        self.label_w_units.grid(row=5, column=5, padx=self.padding, pady=self.padding, sticky='e')
        self.label_s_units = ttk.Label(frame, text="mm")
        self.label_s_units.grid(row=7, column=5, padx=self.padding, pady=self.padding, sticky='e')

        self.button_calc = ttk.Button(frame, text='Calculate Inductance', state=NORMAL)
        self.button_calc.grid(row=9, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.button_calc.config(command=self.callback_calc)

        self.button_clr = ttk.Button(frame, text='Reset')
        self.button_clr.grid(row=9, column=3, padx=self.padding, pady=self.padding, sticky='w')
        self.button_clr.config(command=self.callback_clr)

        self.button_example = ttk.Button(frame, text='Example')
        self.button_example.grid(row=9, column=5, padx=self.padding, pady=self.padding, sticky='w')
        self.button_example.config(command=self.callback_dfts)
        tp.Tooltip(self.button_example,
                   text='Example parameters will be placed into the parameter fields that result in an inductance of 16.5e-06H',
                   wraplength=100)
        tp.Tooltip(self.button_calc,
                   text='Reference: Simple Accurate Expressions for Planar Spiral Inductances, IEEE Journal of Solid-State Circuits, Oct. 1999, pp. 1419-25.',
                   wraplength=100)

        self.entry_n.focus()

    def message_frame(self, frame):
        self.label_message = ttk.Label(frame, textvariable=self.message, relief=FLAT, width=50)
        self.label_message.grid(row=0, column=0, padx=self.padding, pady=self.padding, sticky='w', columnspan=1)

    def calculation_frame(self, frame):
        # dout; outer dimension of coil
        self.label_calc_dout = ttk.Label(frame, text='Calculated  outer diameter (dout):')
        self.label_calc_dout.grid(row=12, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_dout = ttk.Entry(frame, width=20)
        self.entry_calc_dout.grid(row=12, column=3, sticky='w')
        self.label_calc_dout = ttk.Label(frame, text="mm")
        self.label_calc_dout.grid(row=12, column=5, padx=self.padding, pady=self.padding, sticky='e')

        # rho; fill factor
        self.label_calc_rho = ttk.Label(frame, text='Calculated fill factor (rho):')
        self.label_calc_rho.grid(row=14, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_rho = ttk.Entry(frame, width=20)
        self.entry_calc_rho.grid(row=14, column=3, sticky='w')
        tp.Tooltip(self.label_calc_rho, text=u'Fill factor, \u03C1=(dout-din)/(dout+din)', wraplength=200)

        # modified wheeler
        self.label_calc_mw = ttk.Label(frame, text='Modified Wheeler:')
        self.label_calc_mw.grid(row=16, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_mw = ttk.Entry(frame, width=20)
        self.entry_calc_mw.grid(row=16, column=3, sticky='w')
        self.label_calc_mw_units= ttk.Label(frame, text="H")
        self.label_calc_mw_units.grid(row=16, column=5, padx=self.padding, pady=self.padding, sticky='e')
        tp.Tooltip(self.label_calc_mw, text=u'A modification of the original Wheeler formula.', wraplength=200)

        # current sheet
        self.label_calc_cs = ttk.Label(frame, text='Current Sheet:')
        self.label_calc_cs.grid(row=18, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_cs = ttk.Entry(frame, width=20)
        self.entry_calc_cs.grid(row=18, column=3, sticky='w')
        self.label_calc_cs = ttk.Label(frame, text="H")
        self.label_calc_cs.grid(row=18, column=5, padx=self.padding, pady=self.padding, sticky='e')

        # monomial fit
        self.label_calc_mf = ttk.Label(frame, text='Monomial Fit:')
        self.label_calc_mf.grid(row=20, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_mf = ttk.Entry(frame, width=20)
        self.entry_calc_mf.grid(row=20, column=3, sticky='w')
        self.label_calc_mf = ttk.Label(frame, text="H")
        self.label_calc_mf.grid(row=20, column=5, padx=self.padding, pady=self.padding, sticky='e')

        # Fasthenry`
        self.label_calc_fh = ttk.Label(frame, text='Fasthenry:')
        self.label_calc_fh.grid(row=22, column=1, padx=self.padding, pady=self.padding, sticky='w')
        self.entry_calc_fh = ttk.Entry(frame, width=20)
        self.entry_calc_fh.grid(row=22, column=3, sticky='w')
        self.label_calc_fh = ttk.Label(frame, text="H")
        self.label_calc_fh.grid(row=22, column=5, padx=self.padding, pady=self.padding, sticky='e')

    def read_params(self):
        valid = self.validate_args()
        if valid:
            n = float(self.entry_n.get())
            w = float(self.entry_w.get())
            s = float(self.entry_s.get())
            din = float(self.entry_d.get())

            self.dout.set(str(din + 2 * n * w + 2 * (n - 1) * s))
            dout = din + 2 * n * w + 2 * (n - 1) * s
            self.rho.set(str((dout - din) / (dout + din)))
            return True
        else:
            self.message.set('Please check the entered values')
            return False

    def callback_solver(self):
        at_least_one_is_empty = \
            len(self.entry_t_ind.get()) == 0 or \
            len(self.entry_t_rho.get()) == 0

        if not at_least_one_is_empty:
            are_floats = \
                self.r.match(self.entry_t_ind.get()) and \
                self.r.match(self.entry_t_rho.get())
        else:
            self.message.set('Please enter values')
            return False

        if are_floats:
            test = float(self.entry_t_ind.get()) >= 2.0e-09
            test = test and float(self.entry_t_rho.get()) > 0
        else:
            self.message.set('An entered value is not a valid float')
            test = False

        if not at_least_one_is_empty and are_floats and test:
            self.solver()
            return True
        else:
            self.message.set('Please check the entered values')
            return False

    def solver(self):
        self.message.set('Solving...')
        root.update()
        h = self.entry_c.get()
        fh_solver = fh.Fasthenry(0, 0, 0, 0, h)
        fh_solver.runOptimization(self, root)
        self.callback_calc()
        self.message.set('Ready')

    def callback_calc(self):
        valid = self.read_params()
        if valid:
            lmw, lgmd, lmon, lfh = self.calculate_inductance()
            kfp = fp.Footprint(self.din.get(), self.n.get(), self.w.get(), self.s.get(), self.dout.get(),
                               self.rho.get(), lfh)
            footprint = kfp.formKiCadFootprint()
            filename = 'planar_inductor_{:.3f}_{:.3f}_{:.3f}_{}_{:.3e}.kicad_mod'.format(float(self.din.get()),
                                                                                         float(self.w.get()),
                                                                                         float(self.s.get()),
                                                                                         int(self.n.get()),
                                                                                         lfh)
            f = open(filename, 'w+')
            f.write(footprint)
            f.close()

            print('kicad footprint saved')
            root.update()

            self.entry_calc_fh.get()
            self.entry_calc_fh.delete(0, END)
            self.entry_calc_fh.insert(0, '{}'.format(self.eng_string(lfh, sig_figs=6, si=False)))

            self.entry_calc_cs.get()
            self.entry_calc_cs.delete(0, END)
            self.entry_calc_cs.insert(0, '{}'.format(self.eng_string(lgmd, sig_figs=6, si=False)))

            self.entry_calc_mw.get()
            self.entry_calc_mw.delete(0, END)
            self.entry_calc_mw.insert(0, '{}'.format(self.eng_string(lmw, sig_figs=6, si=False)))

            self.entry_calc_mf.get()
            self.entry_calc_mf.delete(0, END)
            self.entry_calc_mf.insert(0, '{}'.format(self.eng_string(lmon, sig_figs=6, si=False)))

            self.entry_calc_dout.get()
            self.entry_calc_dout.delete(0, END)
            self.entry_calc_dout.insert(0, '{:.4f}'.format(float(self.dout.get())))

            self.entry_calc_rho.get()
            self.entry_calc_rho.delete(0, END)
            self.entry_calc_rho.insert(0, '{:.4f}'.format(float(self.rho.get())))

    def callback_clr(self):
        for e in [self.entry_n, self.entry_d, self.entry_w, self.entry_s, self.entry_calc_fh,
                  self.entry_calc_cs, self.entry_calc_dout, self.entry_calc_mf,
                  self.entry_calc_mw, self.entry_calc_rho]:
            e.delete(0, END)
            self.message.set(self.default_message)
            self.entry_n.focus()

    def callback_dfts(self):
        self.n.set(self.dft_n.get())
        self.w.set(self.dft_w.get())
        self.s.set(self.dft_s.get())
        self.din.set(self.dft_din.get())
        self.c.set(self.min_c.get())

    def calculate_inductance(self):
        a = [1.62e-3, -1.21, -0.147, 2.40, 1.78, -0.030]  # values from Table III of paper.
        b = a[0]

        din = float(self.din.get())
        dout = float(self.dout.get())
        davg = np.divide((dout + din), 2)
        n = float(self.n.get())
        w = float(self.w.get())
        s = float(self.s.get())
        h = float(self.min_c.get())
        # Monomial Expression
        lmon = np.multiply(b, np.power(dout, a[1]))
        lmon = np.multiply(lmon, np.power(w, a[2]))
        lmon = np.multiply(lmon, np.power(davg, a[3]))
        lmon = np.multiply(lmon, np.power(n, a[4]))
        lmon = np.multiply(lmon, np.power(s, a[5]))

        # Modified Wheeler
        k1 = 2.34
        k2 = 2.75
        rho = float(self.rho.get())
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

        # fasthenry
        fh_ind = fh.Fasthenry(din, n, w, s, h)
        lfh, message = fh_ind.determine_impedance_of_square_planar_inductor_using_fasthenry()
        self.message.set(message)

        return np.multiply(lmw, 1e-06), np.multiply(lgmd, 1e-06), np.multiply(lmon, 1e-06), np.multiply(lfh, 1)

    def eng_string(self, x, sig_figs=3, si=True):
        """
        Returns float/int value <x> formatted in a simplified engineering format -
        using an exponent that is a multiple of 3.

        sig_figs: number of significant figures

        si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
        e-9 etc.
        """
        x = float(x)
        sign = ''
        if x < 0:
            x = -x
            sign = '-'
        if x == 0:
            exp = 0
            exp3 = 0
            x3 = 0
        else:
            exp = int(math.floor(math.log10(x)))
            exp3 = exp - (exp % 3)
            x3 = x / (10 ** exp3)
            x3 = round(x3, -int(math.floor(math.log10(x3)) - (sig_figs - 1)))
            if x3 == int(x3):  # prevent from displaying .0
                x3 = int(x3)

        if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
            exp3_text = 'yzafpnum kMGTPEZY'[exp3 // 3 + 8]
        elif exp3 == 0:
            exp3_text = ''
        else:
            exp3_text = 'e%s' % exp3

        return ('%s%s%s') % (sign, x3, exp3_text)


root = Tk()
my_gui = Gui(root)
root.mainloop()
