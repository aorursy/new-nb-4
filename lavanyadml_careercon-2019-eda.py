import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from bokeh.plotting import figure, ColumnDataSource

from bokeh.models import HoverTool, LinearColorMapper, ColorBar, FuncTickFormatter, FixedTicker, AdaptiveTicker

from itertools import combinations, product, zip_longest

from scipy.stats import skew, kurtosis, gaussian_kde

from collections import Counter



bar_color = "cornflowerblue"

colors = ["#ADD8E6", "#9AC7E7", "#88B6E9", "#76A5EB", "#6495ED", "#647CD8", "#6564C3", "#654BAE", "#663399"]



def scatter_with_hover(df, x, y,

                       fig=None, cols=None, name=None, marker='x',

                       fig_width=500, fig_height=500, **kwargs):

    """

    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic

    tooltips showing columns from `df`.

    Parameters

    ----------

    df : pandas.DataFrame

        DataFrame containing the data to be plotted

    x : str

        Name of the column to use for the x-axis values

    y : str

        Name of the column to use for the y-axis values

    fig : bokeh.plotting.Figure, optional

        Figure on which to plot (if not given then a new figure will be created)

    cols : list of str

        Columns to show in the hover tooltip (default is to show all)

    name : str

        Bokeh series name to give to the scattered data

    marker : str

        Name of marker to use for scatter plot

    **kwargs

        Any further arguments to be passed to fig.scatter

    Returns

    -------

    bokeh.plotting.Figure

        Figure (the same as given, or the newly created figure)

    Example

    -------

    fig = scatter_with_hover(df, 'A', 'B')

    show(fig)

    fig = scatter_with_hover(df, 'A', 'B', cols=['C', 'D', 'E'], marker='x', color='red')

    show(fig)

    Author

    ------

    Robin Wilson <robin@rtwilson.com>

    with thanks to Max Albert for original code example

    """



    # If we haven't been given a Figure obj then create it with default

    # size etc.

    if fig is None:

        fig = figure(width=fig_width, height=fig_height, tools=['box_zoom', 'reset', 'save'])



    # We're getting data from the given dataframe

    source = ColumnDataSource(data=df)



    # We need a name so that we can restrict hover tools to just this

    # particular 'series' on the plot. You can specify it (in case it

    # needs to be something specific for other reasons), otherwise

    # we just use 'main'

    if name is None:

        name = 'main'



    # Actually do the scatter plot - the easy bit

    # (other keyword arguments will be passed to this function)

    fig.scatter(x=x, y=y, source=source, name=name, marker=marker, **kwargs)



    # Now we create the hover tool, and make sure it is only active with

    # the series we plotted in the previous line

    hover = HoverTool(names=[name])



    if cols is None:

        # Display *all* columns in the tooltips

        hover.tooltips = [(c, '@' + c) for c in df.columns]

    else:

        # Display just the given columns in the tooltips

        hover.tooltips = [(c, '@' + c) for c in cols]



    # Finally add/enable the tool

    fig.add_tools(hover)



    return fig





def block_heatmap(df, height=600, width=900):

    """

    Generates a

    :param df:

        The Pandas DataFrame to render in block-heatmap style.

    :return:

        A Bokeh block heatmap figure modeled after example code.  The figure has additional properties, df for

        the plot data, and rect for the plot object.

    """

    # this colormap blatantly copied from the New York Times.

    colors = ["#ADD8E6", "#9AC7E7", "#88B6E9", "#76A5EB", "#6495ED", "#647CD8", "#6564C3", "#654BAE", "#663399"]

    mapper = LinearColorMapper(palette=colors, low=0, high=1)

    cols = {i: c for (i, c) in enumerate(df.columns)}

    index = {i: r for (i, r) in enumerate(df.index)}

    cols_by_rows = product(enumerate(df.columns), enumerate(df.index))

    data = np.array([[x, y, c, r, df.loc[r, c]] for ((x, c), (y, r)) in cols_by_rows])

    combination_df = pd.DataFrame(data, columns=["gene_id", "sample_id", "gene", "sample", "value"])

    source = ColumnDataSource(combination_df)



    fig = figure(title="Clustered Heatmap", toolbar_location="below", x_range=(0, len(df.columns)),

                 y_range=(0, len(df.index)), tools=["box_zoom", "pan", "reset", "save"], name="heatmap",

                 x_axis_location="above", plot_width=width, plot_height=height, active_drag="box_zoom")

    fig.rect(x="gene_id", y="sample_id", source=source, width=1, height=1,

             fill_color={'field': 'value', 'transform': mapper}, line_color=None)



    fig.grid.grid_line_color = None

    fig.axis.axis_line_color = None

    fig.axis.major_tick_line_color = None

    fig.axis.major_label_text_font_size = "7pt"

    fig.axis.major_label_standoff = 0

    fig.xaxis.major_label_orientation = np.pi / 3



    fig.yaxis.formatter = FuncTickFormatter(code="""

        var labels = %s;

        return labels[tick] || '';

    """ % index)



    fig.xaxis.formatter = FuncTickFormatter(code="""

        var labels = %s;

        return labels[tick] || '';

    """ % cols)



    fig.yaxis.ticker = FixedTicker(ticks=list(index.keys()))

    fig.xaxis.ticker = AdaptiveTicker(mantissas=list(range(10)), min_interval=1, max_interval=5)



    hover = HoverTool(names=["heatmap"])

    hover.tooltips = [

        ('gene', '@gene'),

        ('sample', '@sample'),

        ('percentile', '@value%')

    ]

    fig.add_tools(hover)



    return fig





def plot_histogram(*data, title=None, columns=3):

    def plot_data(d, a):

        if d is None:

            a.axis("off")

            return

        a.hist(d, normed=True, color=bar_color, label=None)

        de = gaussian_kde(d)

        edge = 1

        x = pd.Series(np.linspace(edge * d.min(), d.max() / edge, 100))

        interpolated_y = de(x)

        cumulative = x.apply(lambda v: de.integrate_box_1d(d.min(), v)) * interpolated_y.max()

        a.plot(x, interpolated_y, linestyle='--', color="rebeccapurple", label="PDF")

        a.plot(x, cumulative, linestyle='--', color="dimgray", label="CDF")

        a.fill_between(x, interpolated_y, interpolate=True, color="rebeccapurple", alpha=0.35, zorder=10)

        a.fill_between(x, cumulative, interpolate=True, color="dimgray", alpha=0.125, zorder=15)

        a.set_xlim([x.min(), x.max()])



        a.yaxis.set_ticks_position('none')

        a.yaxis.set_ticklabels([])



    if columns > len(data):

        columns = len(data)

    rows = int(np.ceil(len(data) / columns))



    fig, axes = plt.subplots(rows, columns)



    if columns == 1:

        plot_data(data[0], axes)

        if title:

            axes.set_title(title)

        axes.set_ylabel("Density")

        axes.legend()

    else:

        flat_axes = axes.flatten()

        for d, a in zip_longest(data, flat_axes):

            plot_data(d, a)

        if title:

            for t, a in zip(title, flat_axes):

                a.set_title(t)



    fig.tight_layout()

    return fig





def counter_histogram(labels):

    counts = Counter(labels)

    fig, ax = plt.subplots()

    int_keys = [int(k) for k in counts.keys()]

    ax.bar(int_keys, list(counts.values()), color=bar_color)

    ax.set_xticks(sorted(int_keys))



    k_range = max(counts.keys()) - min(counts.keys())

    max_v = max(counts.values())



    def offset(k, v):

        return (k - k_range * 0.0125, v + max_v * 0.01)



    for (k, v) in counts.items():

        ax.annotate(str(v), offset(k, v))





def add_dummy(dataframe, column_name):

    dummies = pd.get_dummies(dataframe[column_name], prefix="dummy_" + column_name)

    return pd.concat([dataframe, dummies], axis=1)





def filtered_combinations(columns, include_dummies=True, combine_dummies=False):

    def filter_if_dummies(t):

        a, b = t

        a_dummy = a.startswith("dummy_")

        b_dummy = b.startswith("dummy_")

        if not include_dummies and (a_dummy or b_dummy):

            return False

        if a_dummy and b_dummy:

            if combine_dummies:

                a_split = a.split("_")

                b_split = b.split("_")

                if not a_split[1] == b_split[1]:

                    return True

            return False

        return True



    return filter(filter_if_dummies, combinations(columns))





def generate_moment_statistics(data):

    data_skew = skew(data)

    data_kurtosis = kurtosis(data)
import networkx as nx

#import hdbscan

import warnings

from collections import Counter

from bokeh.io import output_notebook, show

from bokeh.models import Range1d

from scipy.stats import gaussian_kde, anderson, skew, kurtosis, gamma, entropy

from itertools import zip_longest, count, cycle

#from helpers import block_heatmap, scatter_with_hover, plot_histogram, counter_histogram, bar_color

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import FastICA, PCA

from sklearn.manifold import MDS, TSNE

from statsmodels.distributions.empirical_distribution import ECDF

from ipywidgets import interact

from matplotlib.colors import ListedColormap


warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (12, 5)

output_notebook()

import os

import time

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import lightgbm as lgb

sns.set()



print("Files in the input folder:")

print(os.listdir("../input"))

train = pd.read_csv('../input/X_train.csv')

test = pd.read_csv('../input/X_test.csv')

y = pd.read_csv('../input/y_train.csv')

sub = pd.read_csv('../input/sample_submission.csv')

print("\nX_train shape: {}, X_test shape: {}".format(train.shape, test.shape))

print("y_train shape: {}, submission shape: {}".format(y.shape, sub.shape))
import numpy as np

import pandas as pd



from bokeh.plotting import figure, ColumnDataSource

from bokeh.models import HoverTool, LinearColorMapper, ColorBar, FuncTickFormatter, FixedTicker, AdaptiveTicker

from itertools import combinations, product, zip_longest

from scipy.stats import skew, kurtosis, gaussian_kde

from collections import Counter
train.head()
test.head(3)
y.head(3)
description = train.describe()

coef_variation = description.loc["std"] / description.loc["mean"]

description.loc["cova"] = coef_variation

description.sort_values(by="cova", axis=1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

minmax_normalized_df = pd.DataFrame(MinMaxScaler().fit_transform(train),

                                    columns=train.columns, index=train.index)



standardized_df = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns,

                               index=train.index)



ecdf_normalized_df = train.apply(lambda c: pd.Series(ECDF(c)(c), index=c.index))
plot_histogram(description.loc["cova"], title="Coefficient of variation");
high_cova = description.loc["cova"].where(lambda x: x > 0.30).dropna().sort_values(ascending=False)

high_cova
high_anderson_and_cova = set(high_cova.index) & set(high_cova.index)

print(high_anderson_and_cova)
series_data = [train[series_id] for series_id in high_anderson_and_cova]



plot_histogram(*series_data, title=high_anderson_and_cova);
train=train.drop(['row_id'], axis=1)

train.head(3)

skews = train.apply(lambda x: skew(x))

skews.name = "skew"

kurts = train.apply(lambda x: kurtosis(x))

kurts.name = "kurtosis"

series_ids = pd.Series([i.split("_")[0] for i in kurts.index], index=kurts.index, name="series_id")

sk_df = pd.concat([skews, kurts, series_ids], axis=1)

fig = scatter_with_hover(sk_df, "skew", "kurtosis", name="skew_kurt", cols=["series_id"])

show(fig, notebook_handle=True);
high_skew_kurt = sk_df.loc[((skews * kurts).abs() >= 5),["skew", "kurtosis"]]

high_skew_kurt
pc = PCA()

reduced_df = pd.DataFrame(pc.fit_transform(ecdf_normalized_df), index=ecdf_normalized_df)



fig, ax = plt.subplots()

total_var = np.sum(pc.explained_variance_)

cumulative_var = np.cumsum(pc.explained_variance_)

ax.plot(cumulative_var / total_var);

dist = gamma(8, 1)

X = np.linspace(0, 30, 100)

Y = [dist.pdf(x) for x in X]



fig, ax = plt.subplots()

ax.plot(X, Y);