import matplotlib
import math


def latexify(fig_width=None, fig_height=None, columns=1):
  """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
  # From https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html

  # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

  # Width and max height in inches for IEEE journals taken from
  # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

  assert columns in [1, 2]

  if fig_width is None:
    fig_width = 3.39 if columns == 1 else 6.9  # width in inches

  if fig_height is None:
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_height = fig_width * golden_mean  # height in inches

  MAX_HEIGHT_INCHES = 8.0
  if fig_height > MAX_HEIGHT_INCHES:
    print("WARNING: fig_height too large:" + fig_height + "so will reduce to" +
          MAX_HEIGHT_INCHES + "inches.")
    fig_height = MAX_HEIGHT_INCHES

  params = {
      "backend": "pdf",
      "text.latex.preamble": r"\usepackage{gensymb}",
      "axes.labelsize": 10,  # fontsize for x and y labels (was 10)
      "axes.titlesize": 10,
      "font.size": 10,  # was 10
      "legend.fontsize": 10,  # was 10
      "xtick.labelsize": 10,
      "ytick.labelsize": 10,
      "text.usetex": True,
      "figure.figsize": [fig_width, fig_height],
      "font.family": "serif",
  }

  matplotlib.rcParams.update(params)
