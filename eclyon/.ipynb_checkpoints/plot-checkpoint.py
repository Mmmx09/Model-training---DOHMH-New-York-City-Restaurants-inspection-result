import re
import IPython
import graphviz
from sklearn.base import ClassifierMixin
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt


def set_plot_sizes(sml: int, med: int, big: int) -> None:
    plt.rc('font', size = sml)          # controls default text sizes
    plt.rc('axes', titlesize = sml)     # fontsize of the axes title
    plt.rc('axes', labelsize = med)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize = sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize = sml)    # legend fontsize
    plt.rc('figure', titlesize = big)   # fontsize of the figure title
    return

    
def draw_tree(
    tree: ClassifierMixin, 
    feature_names: list[str], 
    size: int = 10, 
    ratio: float = 0.6, 
    precision: int = 0,
    ):
    """
    Draws a representation of a random forest in IPython.
    """
    s = export_graphviz(
        tree, 
        out_file = None, 
        feature_names = feature_names, 
        filled = True, 
        special_characters = True, 
        rotate = True, 
        precision = precision,
    )
    # return s
    IPython.display.display(
        graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))
    )
