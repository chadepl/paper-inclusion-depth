

if __name__ == "__main__":
    import numpy as np
    from skimage.draw import disk
    from skimage.segmentation import find_boundaries
    import matplotlib.pyplot as plt

    r = c = 512
    d2 = np.zeros((r, c))
    rr, cc = disk((512 // 2, 512 // 2), 100, shape=(r, c))
    d2[rr, cc] = 1

    d1 = np.zeros((r, c))
    # Case 1
    # rr, cc = disk((r // 2, c // 2), 70, shape=(r, c))
    # Case 2
    # rr, cc = disk(((r // 2) + 30, (c // 2) + 30), 70, shape=(r, c))
    # Case 3
    rr, cc = disk(((r // 2) + 80, (c // 2) + 80), 70, shape=(r, c))
    # Case 4
    # rr, cc = disk(((r // 2) + 130, (c // 2) + 130), 70, shape=(r, c))
    # Case 5
    # rr, cc = disk((r // 2, c // 2), 120, shape=(r, c))
    d1[rr, cc] = 1

    plt.imshow(d1 + d2)
    plt.show()

    b1 = find_boundaries(d1, mode="inner")
    b2 = find_boundaries(d2, mode="inner")

    def in_func(b1, m2):
        in_pos = b1 * m2
        return in_pos.sum()/b1.sum()

    def out_func(b1, m2):
        out_pos = b1 * (1 - m2)
        return out_pos.sum()/b1.sum()

    in12 = in_func(b1, d2)
    in21 = in_func(b2, d1)
    out12 = out_func(b1, d2)
    out21 = out_func(b2, d1)

    print(in12)
    print(in21)
    print(out12)
    print(out21)

    print(np.minimum(in12, in21))
    print(np.minimum(out12, out21))

    print()
    print(f"sum to in: {in12 * out21}")
    print(f"sum to out: {in21 * out12}")
    print()
    print(f"sum to in: {np.minimum(in12, out21)}")
    print(f"sum to out: {np.minimum(in21, out12)}")