import imageio
import numpy as np


def display_numpy_image(img_numpy):
    if img_numpy.shape[2] == 4:
        img_numpy = img_numpy[:, :, : 3]
    if img_numpy.dtype != np.float32 and img_numpy.dtype != np.float64:
        img_numpy = img_numpy / 255.0
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot
    pyplot.imshow(img_numpy, interpolation="nearest")
    pyplot.show()


def load_image(path):
    img = imageio.imread(path)
    return img


def t1():
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png"

    img1 = load_image(p1)
    img2 = load_image(p2)

    c1 = img1[190: 197, 282: 289, :]
    # c2 = img2[190: 197, 150: 157, :]
    c2 = img2[190: 197, 801: 808, :]

    # display_numpy_image(c1)
    # display_numpy_image(c2)

    c1 = c1.reshape([-1, 3])
    c2 = c2.reshape([-1, 3])

    print(c1.shape)
    print(c2.shape)

    m1 = np.mean(c1, axis=0)[np.newaxis, :]
    m2 = np.mean(c2, axis=0)[np.newaxis, :]

    std1 = np.std(c1, axis=0)[np.newaxis, :]
    std2 = np.std(c2, axis=0)[np.newaxis, :]

    res = np.sum((c1 - m1) * (c2 - m2) / (std1 * std2))
    n = c1.ravel().shape[0]
    res /= n
    print(res)


if __name__ == "__main__":
    t1()