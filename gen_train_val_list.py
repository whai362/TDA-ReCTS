import os
import mmcv
import math
import numpy as np
import random
import Polygon as plg
import argparse

np.random.seed(123456)
random.seed(123456)


def get_file_name_list(root):
    file_name_list = [file_name for file_name in mmcv.utils.scandir(root, '.json')]
    return file_name_list


def load_annotation(anns):
    def load_poly(anns):
        polys = []
        labels = []
        for ann in anns:
            if ann['ignore'] == 0:
                assert len(ann['points']) == 8
                polys.append(np.array(ann['points']).reshape((4, 2)))
                labels.append(ann['transcription'])
        return polys, labels

    line_polys, line_labels = load_poly(anns['lines'])
    char_polys, char_labels = load_poly(anns['chars'])

    return line_polys, line_labels, char_polys, char_labels


def get_iou(pD, pG):
    pD = plg.Polygon(np.array(pD))
    pG = plg.Polygon(np.array(pG))
    return get_intersection(pD, pG) / pG.area()


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def center(poly):
    return np.mean(np.array(poly), axis=0)


def distance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.sqrt(np.sum(np.square(c1 - c2)))


def scale(poly):
    return math.sqrt(plg.Polygon(np.array(poly)).area() + 1e-3)


def is_large_character_aspacing(in_chars, thresh):
    centers = [center(char) for char in in_chars]
    n = len(centers)
    if n < 2:
        return False
    mean_dis = []
    for i in range(n):
        min_dis = 1e30
        for j in range(n):
            if i == j:
                continue
            min_dis = min(min_dis, distance(centers[i], centers[j]))
        mean_dis.append(min_dis)
    mean_dis = np.mean(mean_dis)
    mean_scale = np.mean([scale(char) for char in in_chars])
    return mean_dis / mean_scale > thresh


def is_multiple_lines_abreast(i, lines, line_in_chars, thresh):
    if len(line_in_chars[i]) < 2:
        return False
    line_i = np.array(lines[i])
    t_i, l_i = np.min(line_i, 0)
    b_i, r_i = np.max(line_i, 0)
    h_i, w_i = b_i - t_i, r_i - l_i
    in_chars_i = line_in_chars[i]
    scale_i = np.mean([scale(char) for char in in_chars_i])
    for j in range(len(line_in_chars)):
        if i == j:
            continue
        if len(line_in_chars[j]) == 0:
            continue
        line_j = np.array(lines[j])
        t_j, l_j = np.min(line_j, 0)
        b_j, r_j = np.max(line_j, 0)
        in_chars_j = line_in_chars[j]
        scale_j = np.mean([scale(char) for char in in_chars_j])
        s = scale_i / scale_j
        if s <= thresh[1] or s >= 1.0 / thresh[1]:
            continue
        if h_i <= w_i and min(abs(t_i - t_j), abs(b_i - b_j)) < scale_i * 3 and \
                (abs(l_i - l_j) < scale_i * thresh[0] or abs(l_i - l_j) < scale_i * thresh[0]):
            return True
        if h_i > w_i and min(abs(l_i - l_j), abs(r_i - r_j)) < scale_i * 3 and \
                (abs(t_i - t_j) < scale_i * thresh[0] or abs(b_i - b_j) < scale_i * thresh[0]):
            return True
    return False


def assign_char(line, chars):
    in_chars = []
    for char in chars:
        if get_iou(line, char) > 0.7:
            in_chars.append(char)
    return in_chars


def hard_enough(path, thresh):
    lca_thr, mla_thr = thresh[0], thresh[1]
    anns = mmcv.load(path)
    lines, line_labels, chars, char_labels = load_annotation(anns)
    lca_cnt, jtl_cnt = 0, 0
    line_in_chars = []
    for line, line_label in zip(lines, line_labels):
        in_chars = assign_char(line, chars)
        line_in_chars.append(in_chars)

    for i in range(len(line_in_chars)):
        # check large character spacing
        is_lca = is_large_character_aspacing(line_in_chars[i], lca_thr)
        # check juxtaposed text lines
        is_jtl = is_multiple_lines_abreast(i, lines, line_in_chars, mla_thr)

        lca_cnt += int(is_lca)
        jtl_cnt += int(is_jtl)

    if lca_cnt > 0:
        return 1
    elif jtl_cnt > 0:
        return 2
    return 0


def filter_hard_sample(root, file_name_list, thresh):
    hard_list = [[], []]
    for i, file_name in enumerate(file_name_list):
        if i % 1000 == 0:
            print('%d / %d' % (i, len(file_name_list)))

        file_path = root + file_name
        hard_type = hard_enough(file_path, thresh)
        if hard_type > 0:
            file_name = file_name.replace('.json', '')
            hard_list[hard_type - 1].append(file_name)

    np.random.shuffle(hard_list[0])
    np.random.shuffle(hard_list[1])

    return hard_list[0][:500] + hard_list[1][:500]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', nargs='?', type=str)
    args = parser.parse_args()
    if args.data_root is None:
        raise NotImplementedError('Data root is required.')

    rects_train_gt_root = args.data_root + 'train/gt/'
    file_name_list = get_file_name_list(rects_train_gt_root)
    test_list = filter_hard_sample(rects_train_gt_root, file_name_list, [2, (0.1, 0.9)])

    ReCTS_train_list = ''
    for file_name in file_name_list:
        file_name = file_name.replace('.json', '')
        if file_name not in test_list:
            ReCTS_train_list += file_name + '\n'

    list_root = 'train_val_list/'
    if not os.path.exists(list_root):
        os.makedirs(list_root)

    train_list_path = os.path.join(list_root, 'TDA_ReCTS_train_list.txt')
    with open(train_list_path, 'w') as f:
        f.write(ReCTS_train_list)

    ReCTS_val_list = ''
    for file_name in test_list:
        ReCTS_val_list += file_name + '\n'

    val_list_path = os.path.join(list_root, 'TDA_ReCTS_val_list.txt')
    with open(val_list_path, 'w') as f:
        f.write(ReCTS_val_list)
