import time
import matplotlib.pyplot as pt
import numpy as np
from pprint import pprint


class Axis(object):
    def __init__(self, xs, all_ys, title, xlabel, ylabel, labels, is_x_logscale=False, is_y_logscale=False):
        self.xs = xs
        self.all_ys = all_ys
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.labels = labels
        self.title = title
        self.is_x_logscale = is_x_logscale
        self.is_y_logscale = is_y_logscale
        self.colors = [[116, 172, 214], [125, 224, 64], [220, 58, 43], [203, 81, 218], [160, 134, 54], [97, 212, 162],
                       [172, 105, 141], [124, 114, 217], [91, 135, 79], [215, 206, 68], [195, 100, 92], [217, 67, 121],
                       [102, 213, 213], [207, 118, 51], [71, 127, 137], [145, 221, 122], [204, 160, 215], [83, 154, 48],
                       [200, 85, 175], [103, 120, 180]]

    def draw(self):
        for ys, color, label in zip(self.all_ys, self.colors, self.labels):
            if self.is_x_logscale and self.is_y_logscale:
                plot_func = pt.loglog
            elif self.is_x_logscale:
                plot_func = pt.semilogx
            elif self.is_y_logscale:
                plot_func = pt.semilogy
            else:
                plot_func = pt.plot
            if '|' in label:
                marker = label.split('|')[0]
                label = label.split('|')[1]
            else:
                marker = '-^'
            plot_func(self.xs, ys, marker, label=label, color=np.array(color) / 255.0)
        pt.xlabel(self.xlabel)
        pt.ylabel(self.ylabel)
        pt.title(self.title)
        pt.legend(loc=2, borderaxespad=0, bbox_to_anchor=(1.01, 1))


class ExperimentRunner(object):
    def __init__(self, title, xvalues, labels_and_functions, xlabel, ylabels=(), num_repeats=2, num_rounds=2,
                 is_x_logscale=False, is_y_logscale=[False]):
        self.title = title
        self.xlabel = xlabel
        self.ylabels = ['Running time (seconds)'] + list(ylabels)
        self.xvalues = xvalues
        self.labels_and_functions = labels_and_functions
        self.num_repeats = num_repeats
        self.num_rounds = num_rounds
        self.is_x_logscale = is_x_logscale
        self.is_y_logscale = [False] + is_y_logscale

    def run_rounds(self):
        all_ys = [self.run_round() for _ in range(self.num_rounds)]
        all_axes = []
        for ys in zip(*all_ys):
            a = [[sum(values)/float(len(values)) for values in list(zip(*y))] for y in zip(*ys)]
            a = list(zip(*a))
            all_axes.append(a)

        return list(zip(*all_axes))

    def run_round(self):
        ys = []
        for i in range(len(self.labels_and_functions)):
            y = []
            for x in self.xvalues:
                results = []
                for _ in range(self.num_repeats):
                    start_time = time.time()
                    result = self.labels_and_functions[i][1](x)
                    running_time = time.time() - start_time
                    if isinstance(result, tuple):
                        result = list(result)
                    else:
                        result = [result]
                    results.append([running_time] + result)
                y.append([min(res) for res in zip(*results)])
            ys.append(y)
        return ys

    def get_axes(self, ydata, markers=()):
        axes = []
        labels = [l[0] for l in self.labels_and_functions]
        for i, ys in enumerate(ydata):
            a = Axis(self.xvalues, ys, self.title, self.xlabel, self.ylabels[i], labels,
                     is_x_logscale=self.is_x_logscale, is_y_logscale=self.is_y_logscale[i])
            axes.append(a)
        return axes

    def plot_axes(self, ydata):
        axes = self.get_axes(ydata)
        for item in axes:
            pt.figure()
            item.draw()


