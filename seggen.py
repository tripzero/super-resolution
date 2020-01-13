import cv2
import numpy as np

from model import resolve_single
from model.srgan import generator
from utils import load_image


def get_quadrants(img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    y1 = 0
    M = img_height // 2
    N = img_width // 2

    quadrants = []

    for y in range(0, img_height, M):
        for x in range(0, img_width, N):
            y1 = y + M
            x1 = x + N
            tile = img[y:y1, x:x1]
            quadrants.append(tile)

    return quadrants


def assemble_from_quadrants(quadrants):
    img = quadrants[0]

    img_height = img.shape[0]
    img_width = img.shape[1]

    y1 = 0
    M = img_height * 2
    N = img_width * 2

    final_img = np.zeros((M, N, 3), dtype=np.uint8)

    i = 0

    for y in range(0, M, img_height):
        for x in range(0, N, img_width):
            y1 = y + img_height
            x1 = x + img_width
            final_img[y:y1, x:x1] = quadrants[i]
            i += 1

    return final_img


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default='demo/0869x4-crop.png')
    parser.add_argument("--model", default='weights/srgan/gan_generator.h5')

    args = parser.parse_args()

    lr = load_image(args.image)

    model = generator()
    model.load_weights(args.model)

    quads = get_quadrants(lr)

    sr_quads = []

    for quad in quads:
        sr = resolve_single(model, quad)
        sr = resolve_single(model, sr)

        sr_quads.append(sr.numpy())

    final_image = assemble_from_quadrants(sr_quads)

    print(final_image.shape)

    cv2.imwrite("sr.png", resolve_single(model, lr).numpy())
    cv2.imwrite("final.png", final_image)


if __name__ == "__main__":
    main()
