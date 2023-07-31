

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from skimage.io import imread
    import matplotlib.pyplot as plt

    def pin(bi, bj):
        eps_out = (bi - bj)
        eps_out = (eps_out > 0).sum()
        eps_out = eps_out / (bi.sum() + np.finfo(float).eps)
        return 1 - eps_out

    def pout(bi, bj):
        eps_in = (bj - bi)
        eps_in = (eps_in > 0).sum()
        eps_in = eps_in / (bj.sum() + np.finfo(float).eps)
        return 1 - eps_in

    data_dir = Path("../../data/mbod-shapes")

    cases = dict(c1=[1, 2], c2=[1, 3], c3=[2, 4], c4=[2, 1], c5=[5, 6], c6=[5, 7])

    case = 6
    bi_id = cases[f"c{case}"][0]
    bj_id = cases[f"c{case}"][1]

    bi = imread(data_dir.joinpath(f"shape-{bi_id}.png"), as_gray=True)
    bi = (bi > 0).astype(float)
    bj = imread(data_dir.joinpath(f"shape-{bj_id}.png"), as_gray=True)
    bj = (bj > 0).astype(float)

    plt.imshow(bi + bj)
    plt.show()

    print(f"Case {case}: pin: {pin(bi, bj)} | pout: {pout(bi, bj)}")

    # Case 1: pin: 0.5237104300668433 | pout: 1.0
    # Case 2: pin: 0.6077689494261571 | pout: 0.9857325492201483
    # Case 3: pin: 0.920048163756773 | pout: 0.5561337748826376
    # Case 4: pin: 1.0 | pout: 0.5237104300668433
    # Case 5: pin: 0.5829800275833887 | pout: 0.5406698564593302
    # Case 6: pin: 0.0 | pout: 0.0