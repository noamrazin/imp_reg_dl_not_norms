
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def create_line_plot_figure(x_values, y_values, title="", xlabel="", ylabel="", line_labels=None, markers=None, colors=None):
    """
    Creates and returns a figure with line plots for the given x and y values.
    :param x_values: sequence of x value sequences. Each element in x_values should be a sequence of x values for the matching line.
    :param y_values: sequence of y value sequences. Each element in y_values should be a sequence of y values for the matching line.
    :param title: title of the figure.
    :param xlabel: label for x axis.
    :param ylabel: label for y axis.
    :param line_labels: optional names for the lines to be plotted. Length should match x_values and y_values length.
    :param markers: optional marker values for the lines.
    :param colors: optional color values for the lines.
    :return: line plots figure.
    """
    names = line_labels if line_labels is not None else [""] * len(x_values)
    markers = markers if markers is not None else [""] * len(x_values)
    colors = colors if colors is not None else [""] * len(x_values)

    fig = create_garbage_collectable_figure()
    ax = fig.add_subplot(111)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for x, y, name, marker, color in zip(x_values, y_values, names, markers, colors):
        if color:
            ax.plot(x, y, label=name, marker=marker, color=color)
        else:
            ax.plot(x, y, label=name, marker=marker)

    if line_labels is not None:
        ax.legend()

    return fig


def create_garbage_collectable_figure():
    """
    Creates a matplotlib figure object that is garbage collectable and doesn't need to be closed. Can be used for creating plots for saving and not
    showing.
    :return: figure object.
    """
    fig = Figure()
    FigureCanvas(fig)
    return fig
