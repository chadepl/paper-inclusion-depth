"""
This experiment explores what happens with depths
and depth-induced metrics and analyses when part
of the information changes.
With grids, this refers to focusing on one part of
the grid.
With curves, this referes to focusing on parts of the
intervals that are interesting.
More generally, it is about indexing one part of the
data array or removing it from the grid.

Assumptions in this file is that we have clustering
information already.
"""

# We first find SDFs
# We do PCA on SDFs and take the first 2 components
# We then compute depths using Halfspace depths. These are the global depths
# Then we partition the image grid into a GxG grid
# We remove parts of the image and recompute depths
# These are local depths
