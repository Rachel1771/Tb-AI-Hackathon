import os
import cv2
import numpy as np
import onnxruntime as ort
from paddleseg.transforms import Compose, Resize, Normalize
from paddleseg.utils import visualize
from page2 import vis
from PIL import Image
import time
def predict(onnx_model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None):
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    custom_color = [
        0, 0, 0,  # 背景
        242, 94, 141,  # 肝脏
        255, 255, 0  # 肿瘤
    ]
    color_map = vis.get_color_map_list(256, custom_color=custom_color)
    for im_path in image_list:
        # 读取图像
        image = cv2.imread(im_path)
        # ori_shape = image.shape[:2]
        image, _ = transforms(image)
        image = image[np.newaxis, :].astype(np.float32)

        # 运行ONNX模型进行预测
        outputs = ort_session.run(None, {input_name: image})
        segmentation_map = outputs[0]

        # 后处理预测结果
        segmentation_map = np.squeeze(segmentation_map)
        segmentation_map = np.argmax(segmentation_map, axis=0).astype(np.uint8)

        # 可视化预测结果
        added_image = vis.visualize(im_path, segmentation_map, color_map,weight=0.6)
        added_image_path = os.path.join(added_saved_dir, os.path.basename(im_path))
        os.makedirs(added_saved_dir, exist_ok=True)
        cv2.imwrite(added_image_path, added_image)

        # 保存伪彩色的预测结果
        pred_mask = vis.get_pseudo_color_map(segmentation_map,color_map)
        pred_saved_path = os.path.join(pred_saved_dir, os.path.splitext(os.path.basename(im_path))[0] + ".png")
        os.makedirs(pred_saved_dir, exist_ok=True)
        pred_mask.save(pred_saved_path)
        pred_mask_rgb = pred_mask.convert('RGB')
        count = getcount(pred_mask_rgb)
        area = getarea(count)
        return count,area

def getcount(image):
    image.convert('RGB')
    count = 0
    for x in range(image.width):
        for y in range(image.height):
            pixel_value = image.getpixel((x, y))
            if pixel_value == (255, 255, 0):
                count += 1
    return count
def getarea(count):
    area = (count*0.7676*0.7676)/100
    return area

# 调用预测函数
# onnx_model_path = '../model/Unet.onnx'
#
# transforms = Compose([
#     Resize(target_size=(512, 512)),
#     Normalize()
# ])
#
# predict(onnx_model_path, transforms, image_list, save_dir=save_dir)