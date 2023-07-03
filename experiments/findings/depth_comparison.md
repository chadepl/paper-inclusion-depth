# Depth comparison

Also, would it make sense to use the containment data without the threshold?

- We would get per cell how much spill over there is. Closer to 0 is better.
- So then something that is closer to 0 should be closer to 1.
- So we can just do 1 - cell value and use this as our "soft" depth value

Properties of band depth [1]:

1. Affine invariance
2. Maximality at center
3. Monotonicity relative to deepest point
4. Vanishing at infinity

Properties of functional band depth [2]:

5. A
6. B

The contour band depth is sensitive to shape and

It is expensive to compute though because it relies on triplets.
Furthermore, with small sample sizes of highly varying curves most might have depth of 0.
This requires adding a threshold which makes the depth function less sensitive to shape.
For instance, it there is a contour with high-frequency noise in between spaced out contours,
it will likely have a high centrality.
Crucially, contour band depths discards distance information between curves.

Lp depths based on the contours' SDFs are more efficient to compute (only requiring pairwise comparisons)
Furthermore, in many cases like changing scale, position and rotation, they seem to yield similar results to CBD.
Nevertheless, according to the literature, they do not satisfy (1).
Also, potential disadvantages of SDFs could leak into the depth function.
For example ...

## References

- [1] Zuo, Y., & Serfling, R. (2000). General Notions of Statistical Depth Function. The Annals of Statistics, 28(2),
  461–482.
