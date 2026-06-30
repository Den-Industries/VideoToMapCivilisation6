import random
import time

import cv2
import numpy as np
import math

def hexagon_points(center_x, center_y, r):
    angle_offset = np.pi / 6  # Смещение на 30 градусов для вертикального шестиугольника
    points = np.array([
        (int(center_x + np.cos(angle + angle_offset) * (r * 0.99)), int(center_y + np.sin(angle + angle_offset) * r * 0.77))
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False)])
    return points

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask, mode = False):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    if mode:
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    else:
        img_crop[:] = img_overlay_crop + alpha_inv * img_crop

tiles_images = []
tiles_av_clrs = []

mask = cv2.imread("tiles/mask.png", cv2.IMREAD_GRAYSCALE) / 255.0

for i in range(1, 18):
    tiles_images.append(cv2.imread("tiles/" + str(i) + ".png") * mask[0:100, 0:100, np.newaxis])
    img = tiles_images[len(tiles_images) - 1][39:62,25:76]
    A = np.mean(img, axis=(0,1))
    tiles_av_clrs.append(A.astype(int))

draw_dict = [0,1,2,3,2,7,6,7,10,14,10,11,16,13,14,15,16]

mounts_indexes = [4,5,8,9,12]
mounts_tiles = []
for i in range(5):
    mounts_tiles.append([])
    for u in range(6):
        mounts_tiles[i].append(cv2.imread("gori/" + str(i + 1) + "/" + str(u + 1) + ".png", cv2.IMREAD_UNCHANGED))

skali_indexes = [3,6,11,13,15]

skali_tiles = []
for i in range(6):
    skali_tiles.append(cv2.imread("skali/" + str(i + 1) + ".png", cv2.IMREAD_UNCHANGED))


def get_closest_tile(clr):
    closest_index = -1
    closest_dist = 99999
    for i in range(17):
        dist = np.linalg.norm(tiles_av_clrs[i] - clr)
        if dist < closest_dist:
            closest_index = i
            closest_dist = dist
    return closest_index

color_map = np.zeros([32,32,32],dtype=np.uint8)

for x in range(32):
    for y in range(32):
        for z in range(32):
            clr = [x * 8, y * 8, z * 8]
            color_map[x][y][z] = get_closest_tile(clr)

cap = cv2.VideoCapture("shrek.avi")

for i in range(100):
    cap.read()

map_size = 128

output_resolution = [1280, 720]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 24.0, (output_resolution[0], output_resolution[1]))

scale = 1
scaled = False
frame_counter = -1
while cap.isOpened():
    frame_counter += 1
    ret, frame = cap.read()
    if frame is None:
        break
    start = time.time()
    width_value = frame.shape[1] / map_size
    radius = width_value / (3 ** 0.5)
    height_size = math.ceil(((frame.shape[0] / (radius * 3)) * 2.55)) + 3

    grid = np.zeros([height_size, map_size + 1], dtype=int)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if y % 2 == 0 and x == grid.shape[1] - 1:
                continue
            hex_center = np.array([width_value / 2 + x * width_value - (width_value / 2) * (y % 2), radius * 1.5 * y * 0.77])
            frame_crop = frame[max(int(hex_center[1] - radius / 2), 0):int(hex_center[1] + radius / 2),
                                max(int(hex_center[0] - width_value / 2),0):int(hex_center[0] + width_value / 2)]
            av_clr = np.mean(frame_crop, axis=(0,1))
            closest = color_map[int(av_clr[0]/8)][int(av_clr[1]/8)][int(av_clr[2]/8)]#get_closest_tile(av_clr)
            grid[y, x] = closest

    output = np.zeros((output_resolution[1], output_resolution[0], 3), np.uint8)
    width_value = output.shape[1] / map_size
    radius = width_value / (3 ** 0.5)
    if not scaled:
        scaled = True
        scale = width_value / 51
        for i in range(17):
            tiles_images[i] = cv2.resize(tiles_images[i], None, fx=scale, fy=scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale)
        for i in range(5):
            for u in range(6):
                mounts_tiles[i][u] = cv2.resize(mounts_tiles[i][u], None, fx=scale, fy=scale)
        for i in range(6):
            skali_tiles[i] = cv2.resize(skali_tiles[i], None, fx=scale, fy=scale)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                hex_center = [width_value / 2 + x * width_value - (width_value / 2) * (y % 2), radius * 1.5 * y * 0.77]
                overlay_image_alpha(output, tiles_images[draw_dict[grid[y, x]]], int(hex_center[0] - 50 * scale), int(hex_center[1] - 50 * scale), mask)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] != 1:
                hex_center = [width_value / 2 + x * width_value - (width_value / 2) * (y % 2), radius * 1.5 * y * 0.77]
                overlay_image_alpha(output, tiles_images[draw_dict[grid[y, x]]], int(hex_center[0] - 50 * scale), int(hex_center[1] - 50 * scale), mask)
                if grid[y, x] in mounts_indexes:
                    randed = (x * y + y) % 6
                    overlay_image_alpha(output, mounts_tiles[mounts_indexes.index(grid[y, x])][randed][:,:,:3], int(hex_center[0] - 50 * scale),
                                        int(hex_center[1] - 50 * scale), mounts_tiles[mounts_indexes.index(grid[y, x])][randed][:, :, 3] / 255.0, True)
                if grid[y, x] in skali_indexes:
                    dis = [[0,1],[-1,0],[0,-1],[1,-1],[1,0],[1,1]]
                    if y % 2 == 1:
                        for i in range(6):
                            dis[i][0] -= 1
                    for i in range(6):
                        target = [x + dis[i][0], y + dis[i][1]]
                        if target[0] >= 0 and target[0] < grid.shape[1] and target[1] >= 0 and target[1] < grid.shape[0]:
                            if grid[target[1], target[0]] == 1:
                                overlay_image_alpha(output,
                                                    skali_tiles[i][:, :, :3],
                                                    int(hex_center[0] - 50 * scale),
                                                    int(hex_center[1] - 50 * scale),
                                                    skali_tiles[i][:, :, 3] / 255.0, True)
    print("Time per frame ", time.time() - start, " Current index:", frame_counter, "     Done: ", (frame_counter / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) * 100, "%")
    cv2.imshow("1", cv2.resize(output, None, fx=0.5, fy=0.5))
    cv2.imshow("2", cv2.resize(frame, None, fx=0.5, fy=0.5))
    cv2.waitKey(5)
    out.write(output)


out.release()
