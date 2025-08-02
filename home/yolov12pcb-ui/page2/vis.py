import os
import math
from PIL import Image as PILImage
import cv2
import numpy as np
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


# def predict(model,
#             model_path,
#             transforms,
#             image_list,
#             image_dir=None,
#             save_dir='output',
#             aug_pred=False,
#             scales=1.0,
#             flip_horizontal=True,
#             flip_vertical=False,
#             is_slide=False,
#             stride=None,
#             crop_size=None,
#             custom_color=None):
#     utils.utils.load_entire_model(model, model_path)
#     model.eval()
#     nranks = paddle.distributed.get_world_size()
#     local_rank = paddle.distributed.get_rank()
#     if nranks > 1:
#         img_lists = partition_list(image_list, nranks)
#     else:
#         img_lists = [image_list]
#
#     added_saved_dir = os.path.join(save_dir, 'added_prediction')
#     pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
#     custom_color = [
#         0, 0, 0,  # 背景
#         242, 94, 141,  # 肝脏
#         255, 255, 0  # 肿瘤
#     ]
#     logger.info("Start to predict...")
#     progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
#     color_map = get_color_map_list(256, custom_color=custom_color)
#     # print(color_map)
#     with paddle.no_grad():
#         for i, im_path in enumerate(img_lists[local_rank]):
#             im = cv2.imread(im_path)
#             ori_shape = im.shape[:2]
#             im, _ = transforms(im)
#             im = im[np.newaxis, ...]
#             im = paddle.to_tensor(im)
#
#             if aug_pred:
#                 pred, _ = infer.aug_inference(
#                     model,
#                     data['img'],
#                     trans_info=data['trans_info'],
#                     scales=scales,
#                     flip_horizontal=flip_horizontal,
#                     flip_vertical=flip_vertical,
#                     is_slide=is_slide,
#                     stride=stride,
#                     crop_size=crop_size)
#             else:
#                 pred = infer.inference(
#                     model,
#                     im,
#                     ori_shape=ori_shape,
#                     transforms=transforms.transforms,
#                     is_slide=is_slide,
#                     stride=stride,
#                     crop_size=crop_size)
#             pred = paddle.squeeze(pred)
#             pred = pred.numpy().astype('uint8')
#
#             # get the saved name
#             if image_dir is not None:
#                 im_file = im_path.replace(image_dir, '')
#             else:
#                 im_file = os.path.basename(im_path)
#             if im_file[0] == '/' or im_file[0] == '\\':
#                 im_file = im_file[1:]
#
#             # save added image
#             added_image = visualize(
#                 im_path, pred, color_map, weight=0.6)
#             added_image_path = os.path.join(added_saved_dir, im_file)
#             mkdir(added_image_path)
#             cv2.imwrite(added_image_path, added_image)
#
#             # save pseudo color prediction
#             pred_mask = get_pseudo_color_map(pred, color_map)
#             pred_saved_path = os.path.join(
#                 pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
#             mkdir(pred_saved_path)
#             pred_mask.save(pred_saved_path)
#
#             progbar_pred.update(i + 1)


def visualize(image, result, color_map, save_dir=None, weight=0.6):
    color_map1 = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map1).astype("uint8")
    # print(color_map1)
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_pseudo_color_map(pred, color_map=None):
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def paste_images(image_list):
    assert isinstance(image_list,
                      (list, tuple)), "image_list should be a list or tuple"
    assert len(
        image_list) > 1, "The length of image_list should be greater than 1"

    pil_img_list = []
    for img in image_list:
        if isinstance(img, str):
            assert os.path.exists(img), "The image is not existed: {}".format(
                img)
            img = PILImage.open(img)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        pil_img_list.append(img)

    sample_img = pil_img_list[0]
    size = sample_img.size
    for img in pil_img_list:
        assert size == img.size, "The image size in image_list should be the same"

    width, height = sample_img.size
    result_img = PILImage.new(sample_img.mode,
                              (width * len(pil_img_list), height))
    for i, img in enumerate(pil_img_list):
        result_img.paste(img, box=(width * i, 0))

    return result_img
