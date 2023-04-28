# Experiments

We present a pipeline to extract representative candidates from an ensemble of iso-contours.  
This folder contains experiments that demonstrate and validate the different steps of our pipeline.

## Pipeline

Inputs:
 - Ensemble of size N of iso-contours (binary masks).
Outputs:
 - Labeled ensemble of iso-contours. 
   - There are multiple levels based on how many candidates (K) one desires.
   - For each ensemble member we have a global centrality measure.
   - For each ensemble, we assign a label that denotes the group it belongs to. The labels depend on the current level K.
   - For each ensemble member we have a within and between centrality measure for each level.

Process:
 - Signed distance field computation (SDF).
   - Several methods require SDFs so we precompute them for all the ensemble members.
 - Data matrix formation.
   - It can be a data matrix X with a row corresponding to an element.
   - It can be a (dis)similarity matrix D where a row(i)-column(j) combination indicates a relationship between members.
 - Depth calculation. Different depth definitions require different data matrices. Ideally we want to combine the previous step with this one.
 - Clustering. 
   - Based on the data matrix we get a similarity matrix. We can use this matrix for clustering.
   - Furthermore, we can refine the result using the depth informed clustering procedure of [1].
 - Median and outliers calculation.
   - Medians are one possible representative (trimmed means are another).
   - Outliers need to be removed because they introduce noise.
   - Here we compare the quality of these two processes with respect to other methods.


## Evaluation

The argument that we want to make is that we can find better clusters for the task of representative-based ensemble analysis.
This means that we can better determine the number of distinct shapes.
It also means that the resulting clusters are more compact, being able to effectively prune outliers.
How to we show this is true compared to the state-of-the-art?

Axis along which we are better:
 - Time: compared to CBP, we scale better to larger datasets.


## Experiment directory

The list below explains what the goal of each file in this directory.
 - `depth_comparison.py`: we compare the band, lp and sdf depths. We do it in terms of the centrality scores, appearance and computational time for different ensemble sizes.
 - `red_comparison.py`: we assess the utility of relative depths with different depth definitions.
 - `clustering_comparison.py`: in this file we compare depth-based clustering with other clustering methods out there (mainly against hierarchical alternatives).
 - `clustering_pipeline.py`: this file demonstrates our proposed **complete** pipeline and compares it against other pipelines.
 - `band_vs_border_depth`: we compare band depth with the newly introduced border depth. We want to see if results agree.


## References
 - [1] Jörnsten, R. (2004). Clustering and classification based on the L1 data depth. Journal of Multivariate Analysis, 90(1), 67–89. https://doi.org/10.1016/j.jmva.2004.02.013
