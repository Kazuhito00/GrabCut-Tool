#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import copy
import argparse

import cv2 as cv
import numpy as np
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default='image')
    parser.add_argument("--output", type=str, default='output')

    parser.add_argument("--width", help='cap width', type=int, default=512)
    parser.add_argument("--height", help='cap height', type=int, default=512)

    parser.add_argument("--thickness",
                        help='draw thickness',
                        type=int,
                        default=4)

    parser.add_argument("--mask_alpha",
                        help='mask alpha',
                        type=float,
                        default=0.7)

    parser.add_argument("--iteration",
                        help='iteration count',
                        type=int,
                        default=5)

    args = parser.parse_args()

    return args


def mouse_callback(event, x, y, flags, param):
    global debug_image, mask, iteration, thickness, bgd_model, fgd_model
    global grabcut_flag, drawing_mode
    global prev_point

    # 右クリック：前景指定
    if event == cv.EVENT_RBUTTONDOWN:
        drawing_mode = 1
        prev_point = (x, y)
    elif event == cv.EVENT_RBUTTONUP:
        if drawing_mode == 1:
            drawing_mode = 0
            grabcut_flag = True
    # 左クリック：後景指定
    if event == cv.EVENT_LBUTTONDOWN:
        drawing_mode = 2
        prev_point = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        if drawing_mode == 2:
            drawing_mode = 0
            grabcut_flag = True
    # ドラッグ：描画
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing_mode == 1:
            color = (255, 0, 0)
            mask_balue = 1
        if drawing_mode == 2:
            color = (0, 0, 255)
            mask_balue = 0

        if drawing_mode > 0:
            cv.line(debug_image, (x, y), (prev_point[0], prev_point[1]), color,
                    thickness)
            cv.line(mask, (x, y), (prev_point[0], prev_point[1]), mask_balue,
                    thickness)
            prev_point = (x, y)


def execute_grabcut(image, mask, bgd_model, fgd_model, iteration, roi=None):
    image_width, image_height = image.shape[1], image.shape[0]

    # 処理中表示
    loading_image = copy.deepcopy(image)
    loading_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    loading_image = loading_image * loading_mask[:, :, np.newaxis]
    loading_image = cv.addWeighted(loading_image, 0.7, image, 0.3, 0)
    cv.putText(loading_image, "PROCESSING...",
               (int(image_width / 2) - (6 * 18), int(image_height / 2)),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv.LINE_AA)
    cv.putText(loading_image, "PROCESSING...",
               (int(image_width / 2) - (6 * 18), int(image_height / 2)),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv.LINE_AA)
    cv.imshow(window_name, loading_image)
    _ = cv.waitKey(1)

    # GrabCut実施
    if roi is not None:
        mask, bgd_model, fgd_model = cv.grabCut(image, mask, roi, bgd_model,
                                                fgd_model, iteration,
                                                cv.GC_INIT_WITH_RECT)
    else:
        mask, bgd_model, fgd_model = cv.grabCut(image, mask, None, bgd_model,
                                                fgd_model, iteration,
                                                cv.GC_INIT_WITH_MASK)

    # デバッグ用マスク重畳画像
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    debug_image = image * mask2[:, :, np.newaxis]
    debug_image = cv.addWeighted(debug_image, mask_alpha, image, mask_beta, 0)

    return mask, bgd_model, fgd_model, debug_image


def save_index_color_png(output_path, filename, mask_image):
    # ファイル名(拡張子無し)取得
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # 保存先パス作成
    save_path = os.path.join(output_path, base_filename + '.png')
    print(save_path)

    # インデックスカラーモードで保存
    # 読み込む場合は下記のように記述
    # pil_image = Image.open("sample.png")
    # mask = np.asarray(pil_image)
    color_palette = [0, 0, 0, 255, 255, 255]
    with Image.fromarray(mask_image, mode="P") as png_image:
        png_image.putpalette(color_palette)
        png_image.save(save_path)


if __name__ == '__main__':
    # 描画フラグ
    drawing_mode = 0
    grabcut_flag = False
    prev_point = None

    # ウィンドウ名
    window_name = "GrabCut Sample"
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # 引数解析 #################################################################
    args = get_args()

    input_path = args.input
    output_path = args.output

    width = args.width
    height = args.height

    iteration = args.iteration

    mask_alpha = args.mask_alpha
    mask_beta = 1 - mask_alpha

    thickness = args.thickness

    # ファイル読み込み ##########################################################
    files = glob.glob(os.path.join(input_path, '*'))
    file_index = 0

    # 動作モード ###############################################################
    SELECT_MODE = 0
    ROI_MODE = 1
    GRABCUT_MODE = 2

    mode = SELECT_MODE

    while (True):
        if mode == SELECT_MODE:
            image = cv.imread(files[file_index])
            resize_image = cv.resize(image, (width, height))
            debug_image = copy.deepcopy(resize_image)
            mask = None

            cv.putText(debug_image, "n:Next p:Previous Enter:Confirm select",
                       (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                       2, cv.LINE_AA)
            cv.putText(debug_image, "n:Next p:Previous Enter:Confirm select",
                       (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                       cv.LINE_AA)
        if mode == ROI_MODE:
            image = cv.imread(files[file_index])
            resize_image = cv.resize(image, (width, height))
            roi_image = cv.resize(image, (width, height))

            # 初期ROI選択 #######################################################
            cv.putText(roi_image, "Select ROI and press Enter", (5, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                       cv.LINE_AA)
            cv.putText(roi_image, "Select ROI and press Enter", (5, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)

            roi = cv.selectROI(window_name, roi_image, showCrosshair=False)

            mask = np.zeros(resize_image.shape[:2], dtype=np.uint8)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)

            # GrabCut実施
            mask, bgd_model, fgd_model, debug_image = execute_grabcut(
                resize_image,
                mask,
                bgd_model,
                fgd_model,
                iteration,
                roi,
            )

            # マウスコールバック登録
            cv.setMouseCallback(window_name, mouse_callback)

            # マスク画像保存
            save_mask = np.where((mask == 2) | (mask == 0), 0,
                                 1).astype('uint8')
            save_index_color_png(output_path, files[file_index], save_mask)

            mode = GRABCUT_MODE
        if mode == GRABCUT_MODE and grabcut_flag:
            # マウス描画後GrabCut処理実施
            grabcut_flag = False
            mask, bgd_model, fgd_model, debug_image = execute_grabcut(
                resize_image,
                mask,
                bgd_model,
                fgd_model,
                iteration,
            )

            # マスク画像保存
            save_mask = np.where((mask == 2) | (mask == 0), 0,
                                 1).astype('uint8')
            save_index_color_png(output_path, files[file_index], save_mask)

        # 画面反映 #############################################################
        cv.imshow(window_name, debug_image)
        if mask is not None:
            mask_image = np.where((mask == 2) | (mask == 0), 0,
                                  255).astype('uint8')
            cv.imshow('Mask Image', mask_image)

        # キー処理(ESC：終了、n：次画像へ、p：前画像へ、Enter：画像選択確定) ######
        key = cv.waitKey(1)
        if key == 110:  # n
            if (file_index + 1) < len(files):
                file_index = file_index + 1
                mode = SELECT_MODE
        if key == 112:  # p
            if 0 <= (file_index - 1):
                file_index = file_index - 1
                mode = SELECT_MODE
        if key == 13:  # Enter
            if mode == SELECT_MODE:
                mode = ROI_MODE
        if key == 27:  # ESC
            break

    cv.destroyAllWindows()
