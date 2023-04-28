
import numpy as np
from skimage.draw import ellipse
from skimage.segmentation import find_boundaries
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def get_nestedness(c1, c2):
    bc1 = find_boundaries(c1, mode="inner")
    bc1_id = np.where(bc1)
    bc2 = find_boundaries(c2, mode="inner")
    bc2_id = np.where(bc2)

    c1_in_c2 = c2[bc1].sum() / bc1_id[0].size
    c1_out_c2 = (1 - c2[bc1]).sum() / bc1_id[0].size
    c2_in_c1 = c1[bc2].sum() / bc2_id[0].size
    c2_out_c1 = (1 - c1[bc2]).sum() / bc2_id[0].size

    nestedness = 1 - max(min(c1_in_c2, c2_in_c1), min(c1_out_c2, c2_out_c1))

    return nestedness, c1_in_c2, c1_out_c2, c2_in_c1, c2_out_c1


def create_anim(img_frames, nestedness_scores, anim_name="tmp", title=""):
    frames = []
    fig, axs = plt.subplots(ncols=2, layout="tight", figsize=(10, 5))
    fig.suptitle(title)
    axs[0].set_axis_off()
    axs[1].set_xlim(0, len(img_frames))
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("Animation frame")
    axs[1].set_ylabel("Score")
    axs[1].set_title("Nestedness score: \n 1 - max(min(c1_in, c2_in), min(c1_out, c2_out))")
    for i, img in enumerate(img_frames):
        scores.append(nestedness)
        ar1 = axs[0].imshow(img)
        ar2, = axs[1].plot(np.arange(i+1), np.array(nestedness_scores[:i+1]), c="orange")
        frames.append([ar1, ar2])

    anim = ani.ArtistAnimation(fig, frames, interval=200)
    anim.save(f"/Users/chadepl/Downloads/{anim_name}.mp4")


num_frames = 50

rr, cc = ellipse(100, 100, 70, 70, shape=(200, 400))
c1 = np.zeros((200, 400))
c1[rr, cc] = 1


# Outside, inside
scores = []
imgs = []
for delta in np.linspace(0, 145, num_frames):

    rr, cc = ellipse(100, 100 + delta, 65, 65, shape=(200, 400))
    c2 = np.zeros((200, 400))
    c2[rr, cc] = 1

    nestedness, c1_in_c2, c1_out_c2, c2_in_c1, c2_out_c1 = get_nestedness(c1, c2)

    scores.append(nestedness)
    imgs.append(c1 + c2)

create_anim(img_frames=imgs,
            nestedness_scores=scores,
            anim_name="nest-translation",
            title="Change in nestedness score (translation)")

# Shape deviation
scores = []
imgs = []
for amplitude in np.linspace(0.4, 0, num_frames):

    thetas = np.linspace(0, 2*np.pi, 1000)
    radii = 1 + amplitude * np.sin(10 * thetas)
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    xs = xs * 70 + 100
    ys = ys * 70 + 100
    polygon = np.array([xs, ys]).T

    c2 = polygon2mask((200, 400), polygon)

    nestedness, c1_in_c2, c1_out_c2, c2_in_c1, c2_out_c1 = get_nestedness(c1, c2)

    scores.append(nestedness)
    imgs.append(c1 + c2)

create_anim(img_frames=imgs,
            nestedness_scores=scores,
            anim_name="nest-shape_deviation",
            title="Change in nestedness score (amplitude around radius)")


# Shape deviation
out = False
scores = []
imgs = []
for fact_mult in np.linspace(0, 0.4, num_frames):

    thetas = np.linspace(0, 2*np.pi, 1000)
    sin_factor = np.sin(10 * thetas)
    if out:
        sin_factor[np.where(sin_factor > 0)] = sin_factor[np.where(sin_factor > 0)] * fact_mult
        sin_factor[np.where(sin_factor <= 0)] = sin_factor[np.where(sin_factor <= 0)] * 0.1
    else:
        sin_factor[np.where(sin_factor <= 0)] = sin_factor[np.where(sin_factor <= 0)] * fact_mult
        sin_factor[np.where(sin_factor > 0)] = sin_factor[np.where(sin_factor > 0)] * 0.1
    radii = 1 + sin_factor
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    xs = xs * 70 + 100
    ys = ys * 70 + 100
    polygon = np.array([xs, ys]).T

    c2 = polygon2mask((200, 400), polygon)

    nestedness, c1_in_c2, c1_out_c2, c2_in_c1, c2_out_c1 = get_nestedness(c1, c2)

    scores.append(nestedness)
    imgs.append(c1 + c2)

create_anim(img_frames=imgs,
            nestedness_scores=scores,
            anim_name=f"nest-asym_deviation-out_{out}",
            title="Change in nestedness score (radius)")