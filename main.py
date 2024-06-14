import cv2
import numpy as np
from time import time
import os
import pickle
import pathlib

# Задаем центр шестиугольника и его радиус
radius1, radius2 = 29.44, 25.5
center_x, center_y = -10, 1
Yscale = 0.77

# Функция для получения координат вершин шестиугольника
def hexagon_points(center_x, center_y, radius):
    angle_offset = np.pi / 6  # Смещение на 30 градусов для вертикального шестиугольника
    points = [
        (int(center_x + np.cos(angle + angle_offset) * (radius * 0.99)), int(center_y + np.sin(angle + angle_offset) * radius * Yscale))
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ]
    return points

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    #print(alpha.shape, img_overlay_crop.shape, alpha_inv.shape, img_crop.shape)
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

cap = cv2.VideoCapture("never.mp4") #PUT INPUT IMAGE HERE

images = []

for i in range(17):
    images.append(cv2.imread(str(i + 1) + ".png"))

black = np.zeros((360, 480, 3), dtype="uint8")

#for i in range(120):
#    ret, frame = cap.read()

mountgrid = []
for i in range(30):
    mountgrid.append([0] * 32)
mountatinsgridnum = []
for i in range(5):
    mountatinsgridnum.append(mountgrid)

mountatinsgridnum = np.array(mountatinsgridnum)

if os.path.isfile("computedmountain.bin"):
    f = open("computedmountain.bin", "rb")
    binstuff = f.read()
    f.close()
    mountatinsgridnum = pickle.loads(binstuff)
else:
    for i in range(5):
        print("Processing ", i)
        output = np.zeros((1080, 1920, 3), np.uint8)
        currentgrid = []
        for u in range(30):
            currentgrid.append([0] * 32)
        currentgrid = np.array(currentgrid)
        for x in range(30):
            for y in range(32):
                need = [5, 6, 9, 10, 13]
                currentgrid[x, y] = need[i]
                if currentgrid[x, y] != 2:
                    neighbours = []
                    if y - 1 >= 0 and x - 1 * (1 - y % 2) >= 0:
                        neighbours.append(("ul", (x - 1 * (1 - y % 2), y - 1)))
                    if y - 1 >= 0 and x - 1 * (1 - y % 2) + 1 < 30:
                        neighbours.append(("ur", (x - 1 * (1 - y % 2) + 1, y - 1)))
                    if x - 1 >= 0:
                        neighbours.append(("l", (x - 1, y)))
                    if x + 1 < 30:
                        neighbours.append(("r", (x + 1, y)))
                    if y + 1 < 30 and x - 1 * (1 - y % 2) >= 0:
                        neighbours.append(("dl", (x - 1 * (1 - y % 2), y + 1)))
                    if y + 1 < 30 and x - 1 * (1 - y % 2) + 1 < 30:
                        neighbours.append(("dr", (x - 1 * (1 - y % 2) + 1, y + 1)))

                    hex_points = hexagon_points(center_x + radius2 * 2 * x + radius2 * (y % 2),
                                                center_y + radius1 * 1.5 * y * Yscale, radius1)

                    hex_points = np.array(hex_points)

                    imagenum = currentgrid[x, y]

                    if currentgrid[x, y] == 5:
                        imagenum = 3
                    if currentgrid[x, y] == 6:
                        imagenum = 8
                    if currentgrid[x, y] == 9:
                        imagenum = 11
                    if currentgrid[x, y] == 10:
                        imagenum = 15
                    if currentgrid[x, y] == 13:
                        imagenum = 17

                    image = images[imagenum - 1]

                    height, width, channels = image.shape

                    mask1 = np.zeros((height, width, 1), np.uint8)
                    vertices = np.array([hex_points], np.int32)
                    cv2.fillPoly(mask1, pts=[vertices], color=(255, 255, 255))

                    res2 = cv2.bitwise_and(image, image, mask=mask1)

                    cv2.fillPoly(output, pts=[vertices], color=(0, 0, 0))
                    output = cv2.add(output, res2)

                    if currentgrid[x, y] == 5 or currentgrid[x, y] == 6 or currentgrid[x, y] == 9 \
                            or currentgrid[x, y] == 10 or currentgrid[x, y] == 13:
                        whatindex = [5, 6, 10, 9, 13]
                        bufoutput = output.copy()
                        image1 = images[currentgrid[x, y] - 1]
                        res2 = cv2.bitwise_and(image1, image1, mask=mask1)
                        cv2.fillPoly(bufoutput, pts=[vertices], color=(0, 0, 0))
                        bufoutput = cv2.add(bufoutput, res2)

                        theindex = whatindex.index(currentgrid[x, y])
                        theminimest = -1
                        theminimest_val = 99999

                        x9, y9 = (hex_points[0] + hex_points[3]) / 2
                        x9 -= 50
                        y9 -= 50
                        x9 = int(x9)
                        y9 = int(y9)

                        for tile_c in range(6):
                            img_overlay_rgba = cv2.imread("gori/" + str(theindex + 1) + "/" +
                                                          str(tile_c + 1) + ".png", cv2.IMREAD_UNCHANGED)

                            # Perform blending
                            alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
                            img_result = output[:, :, :3].copy()
                            img_overlay = img_overlay_rgba[:, :, :3]
                            overlay_image_alpha(img_result, img_overlay, x9, y9, alpha_mask)
                            errorL2 = cv2.norm(bufoutput, img_result, cv2.NORM_L2, mask=mask1)
                            if errorL2 < theminimest_val:
                                theminimest_val = errorL2
                                theminimest = tile_c
                        if theminimest == -1:
                            print("WHATA HEEEL")
                            exit()
                        mountatinsgridnum[i, x, y] = theminimest + 1
    obj = pickle.dumps(mountatinsgridnum)
    f = open("computedmountain.bin", "wb")
    f.write(obj)
    f.close()


mountstiles = []
for i in range(5):
    mountstiles.append([])
    for u in range(6):
        mountstiles[i].append(cv2.imread("gori/" + str(i + 1) + "/" + str(u + 1) + ".png", cv2.IMREAD_UNCHANGED))


skalitiles = []
for i in range(6):
    skalitiles.append(cv2.imread("skali/" + str(i + 1) + ".png", cv2.IMREAD_UNCHANGED))

timeforframe = time()

frame_counter = -1
while cap.isOpened():
    frame_counter += 1
    ret, frame = cap.read()

    outputed_filename = "output/" + str(frame_counter) + ".png"
    if os.path.isfile(outputed_filename):
        continue

    currentgrid = []
    for i in range(30):
        currentgrid.append([0] * 32)
    currentgrid = np.array(currentgrid)

    output = np.zeros((1080, 1920, 3), np.uint8)
    start = time()
    for u in range(0, 32):
        for j in range(0, 30):
            mask = np.zeros((360, 480, 1), np.uint8)

            hex_points = hexagon_points(center_x + radius2 * 2 * j + radius2 * (u % 2), center_y + radius1 * 1.5 * u * Yscale, radius1)

            hex_points = np.array(hex_points)

            hex_points_apple = hex_points.copy()
            for b in range(len(hex_points)):
                hex_points_apple[b][0] /= 3
                hex_points_apple[b][1] /= 3

            vertices = np.array([hex_points_apple], np.int32)

            cv2.fillPoly(mask, pts=[vertices], color=(255, 255, 255))

            res = cv2.bitwise_and(frame, frame, mask=mask)

            count = np.count_nonzero(res)

            imagenum = 1

            if count != 0:
                errorL2_1 = cv2.norm(black, res, cv2.NORM_L2, mask=mask)
                res1 = res.copy()
                cv2.fillPoly(res1, pts=[vertices], color=(255, 255, 255))
                errorL2_2 = cv2.norm(black, res1, cv2.NORM_L2, mask=mask)
                bright = errorL2_1 / errorL2_2
                imagenum = int(bright * 16) + 1

            currentgrid[j][u] = imagenum

    for x in range(30):
        for y in range(32):
            neighbours = []
            if x - 1 >= 0:
                neighbours.append(("l", (x - 1, y)))
            if y - 1 >= 0 and x - 1 * (1 - y % 2) >= 0:
                neighbours.append(("ul", (x - 1 * (1 - y % 2), y - 1)))
            if y - 1 >= 0 and x - 1 * (1 - y % 2) + 1 < 30:
                neighbours.append(("ur", (x - 1 * (1 - y % 2) + 1, y - 1)))
            if x + 1 < 30:
                neighbours.append(("r", (x + 1, y)))
            if y + 1 < 30 and x - 1 * (1 - y % 2) >= 0:
                neighbours.append(("dl", (x - 1 * (1 - y % 2), y + 1)))
            if y + 1 < 30 and x - 1 * (1 - y % 2) + 1 < 30:
                neighbours.append(("dr", (x - 1 * (1 - y % 2) + 1, y + 1)))

            if currentgrid[x, y] > 2:
                for neighbour in neighbours:
                    if currentgrid[neighbour[1]] == 1:
                        currentgrid[neighbour[1]] = 2

    for x in range(30):
        for y in range(32):
            if currentgrid[x, y] == 2:
                hex_points = hexagon_points(center_x + radius2 * 2 * x + radius2 * (y % 2),
                                            center_y + radius1 * 1.5 * y * Yscale, radius1)

                hex_points = np.array(hex_points)

                image = images[currentgrid[x, y] - 1]

                height, width, channels = image.shape

                mask1 = np.zeros((height, width, 1), np.uint8)
                vertices = np.array([hex_points], np.int32)
                cv2.fillPoly(mask1, pts=[vertices], color=(255, 255, 255))

                res2 = cv2.bitwise_and(image, image, mask=mask1)

                cv2.fillPoly(output, pts=[vertices], color=(0, 0, 0))
                output = cv2.add(output, res2)

    for x in range(30):
        for y in range(32):
            if currentgrid[x, y] != 2:
                neighbours = []
                if y - 1 >= 0 and x - 1 * (1 - y % 2) >= 0:
                    neighbours.append(("ul", (x - 1 * (1 - y % 2), y - 1)))
                if y - 1 >= 0 and x - 1 * (1 - y % 2) + 1 < 30:
                    neighbours.append(("ur", (x - 1 * (1 - y % 2) + 1, y - 1)))
                if x - 1 >= 0:
                    neighbours.append(("l", (x - 1, y)))
                if x + 1 < 30:
                    neighbours.append(("r", (x + 1, y)))
                if y + 1 < 30 and x - 1 * (1 - y % 2) >= 0:
                    neighbours.append(("dl", (x - 1 * (1 - y % 2), y + 1)))
                if y + 1 < 30 and x - 1 * (1 - y % 2) + 1 < 30:
                    neighbours.append(("dr", (x - 1 * (1 - y % 2) + 1, y + 1)))

                hex_points = hexagon_points(center_x + radius2 * 2 * x + radius2 * (y % 2),
                                            center_y + radius1 * 1.5 * y * Yscale, radius1)

                hex_points = np.array(hex_points)

                imagenum = currentgrid[x, y]

                if currentgrid[x, y] == 5:
                    imagenum = 3
                if currentgrid[x, y] == 6:
                    imagenum = 8
                if currentgrid[x, y] == 9:
                    imagenum = 11
                if currentgrid[x, y] == 10:
                    imagenum = 15
                if currentgrid[x, y] == 13:
                    imagenum = 17

                image = images[imagenum - 1]

                height, width, channels = image.shape

                mask1 = np.zeros((height, width, 1), np.uint8)
                vertices = np.array([hex_points], np.int32)
                cv2.fillPoly(mask1, pts=[vertices], color=(255, 255, 255))


                res2 = cv2.bitwise_and(image, image, mask=mask1)

                cv2.fillPoly(output, pts=[vertices], color=(0, 0, 0))
                output = cv2.add(output, res2)

                if currentgrid[x, y] == 5 or currentgrid[x, y] == 6 or currentgrid[x, y] == 9 \
                        or currentgrid[x, y] == 10 or currentgrid[x, y] == 13:
                    whatindex = [5, 6, 10, 9, 13]
                    theindex = whatindex.index(currentgrid[x, y])
                    need = [5, 6, 9, 10, 13]

                    x9, y9 = (hex_points[0] + hex_points[3]) / 2
                    x9 -= 50
                    y9 -= 50
                    x9 = int(x9)
                    y9 = int(y9)

                    img_overlay_rgba = mountstiles[theindex][mountatinsgridnum[need.index(currentgrid[x, y]), x, y] - 1]

                    # Perform blending
                    alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
                    img_result = output[:, :, :3].copy()
                    img_overlay = img_overlay_rgba[:, :, :3]
                    overlay_image_alpha(img_result, img_overlay, x9, y9, alpha_mask)

                    output = img_result

                if currentgrid[x, y] == 4 or currentgrid[x, y] == 7 or currentgrid[x, y] == 12 or currentgrid[x, y] == 14:
                    for neighbour in neighbours:
                        if currentgrid[neighbour[1]] == 2:
                            to_image_number = ["dl", "l", "ul", "ur", "r", "dr"]
                            #img_overlay_rgba = cv2.imread(
                            #    "skali/" + str(to_image_number.index(neighbour[0]) + 1) + ".png", cv2.IMREAD_UNCHANGED)
                            img_overlay_rgba = skalitiles[to_image_number.index(neighbour[0])]

                            x9, y9 = (hex_points[0] + hex_points[3]) / 2
                            x9 -= 50
                            y9 -= 50
                            x9 = int(x9)
                            y9 = int(y9)

                            # Perform blending
                            alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
                            img_result = output[:, :, :3].copy()
                            img_overlay = img_overlay_rgba[:, :, :3]
                            overlay_image_alpha(img_result, img_overlay, x9, y9, alpha_mask)
                            output = img_result


        #print("Progress ", 100 * ((x + 1) / 32), "%")
    print(time() - timeforframe, " seconds per last frame     Current index:", frame_counter, "     Done: ", (frame_counter / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) * 100, "%")
    timeforframe = time()

    cv2.imwrite("output/" + str(frame_counter) + ".png", output)
    resized = cv2.resize(output, (int(1920 / 3), int(1080 / 3)), interpolation=cv2.INTER_AREA)
    cv2.imshow('Image', resized)
    cv2.imshow("apple", frame)
    cv2.waitKey(1)