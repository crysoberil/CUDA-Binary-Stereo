import imageio
import numpy as np
import skimage.transform
import scipy.misc
import cv2


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
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_gpu.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_gpu (copy).png"

    img1 = load_image(p1)
    img2 = load_image(p2)

    display_numpy_image(img1)
    display_numpy_image(img2)

    exit()

    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small.png"

    img1 = load_image(p1)
    img2 = load_image(p2)

    display_numpy_image(img1)
    display_numpy_image(img2)
    exit()

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


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def reshape_images(f=2):
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png"
    p1_reshaped_path = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small.png"
    p2_reshaped_path = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small.png"
    img1 = load_image(p1)
    img2 = load_image(p2)
    h, w = img1.shape[: 2]
    h /= f
    w /= f
    print(img1.dtype)
    reshaped_img1 = skimage.transform.resize(img1, (h, w), order=1, preserve_range=False)
    reshaped_img2 = skimage.transform.resize(img2, (h, w), order=1, preserve_range=False)
    reshaped_img1 = (reshaped_img1 * 255.0 + 0.5).astype(dtype=np.uint8)
    reshaped_img2 = (reshaped_img2 * 255.0 + 0.5).astype(dtype=np.uint8)
    # display_numpy_image(reshaped_img1)
    scipy.misc.imsave(p1_reshaped_path, reshaped_img1)
    scipy.misc.imsave(p2_reshaped_path, reshaped_img2)


def test_disparity():
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small.png"
    img1, img2 = load_image(p1), load_image(p2)
    stereo = cv2.StereoBM_create(numDisparities=255, blockSize=15)
    disparity = stereo.compute(img1[:,:,0], img2[:,:,0])
    gray = np.concatenate([disparity[:,:,np.newaxis], disparity[:,:,np.newaxis], disparity[:,:,np.newaxis]], axis=2)
    # print(gray.shape)
    # exit()
    display_numpy_image(gray)


def inv_sqrt_test():
    def f(n, iter=5):
        x = 0.5
        for i in range(iter):
            x -= (x * x - 1.0 / n) / (2 * x)
        return x

    import math
    n = 13

    v1 = 1.0 / math.sqrt(n);
    v2 = f(n)

    print(v1)
    print(v2)


def dynamic_test():
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small.png"
    img = load_image(p1)
    img = img[:5, :6, 0] / 255.0
    print(img)


def get_ncc(p1, p2):
    m1 = np.mean(p1, axis=(0, 1))[np.newaxis, np.newaxis, :]
    m2 = np.mean(p2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    inv_std1 = 1.0 / np.std(p1, axis=(0, 1))[np.newaxis, np.newaxis, :]
    inv_std2 = 1.0 / np.std(p2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    terms = (p1 - m1) * (p2 - m2) * (inv_std1 * inv_std2)
    terms = terms.ravel()
    return terms.sum() / terms.shape[0]


def ncc_test(r, c1, c2):
    p1 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small.png"
    p2 = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small.png"

    img1 = load_image(p1) / 255.0
    img2 = load_image(p2) / 255.0

    hf_w_h, hf_w_w = 1, 3

    img1 = img1[r - hf_w_h: r + hf_w_h + 1, c1 - hf_w_w: c1 + hf_w_w + 1, :]
    img2 = img2[r - hf_w_h: r + hf_w_h + 1, c2 - hf_w_w: c2 + hf_w_w + 1, :]

    print(' '.join(["{:.06}".format(elm) for elm in img2[:, :, 0].ravel().tolist()]))

    # display_numpy_image(img1)
    # display_numpy_image(img2)


    ncc = get_ncc(img1, img2)
    print(ncc)



if __name__ == "__main__":
    # t1()
    # reshape_images()
    # test_disparity()
    # inv_sqrt_test()
    # dynamic_test()
    ncc_test(170, 286, 191)