import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCBInference:
    def __init__(self, model_path="best.engine", device="0"):
        """
        初始化推理器
        
        Args:
            model_path (str): TensorRT模型文件路径
            device (str): 设备选择，"0" 表示第一块GPU
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载TensorRT模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            logger.info(f"正在加载TensorRT模型: {self.model_path}")
            
            # 显式指定设备和优化参数
            self.model = YOLO(self.model_path)
            
            # 预热模型 - 这很重要！
            logger.info("正在预热模型...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, device=self.device, verbose=False)
            
            logger.info("模型加载和预热完成!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def preprocess_image(self, image_input, max_size=1920):
        """
        预处理图片，优化尺寸
        
        Args:
            image_input: 输入图片
            max_size (int): 最大边长限制
            
        Returns:
            PIL.Image: 处理后的图片
        """
        # 处理输入图片
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif hasattr(image_input, 'read'):
            image = Image.open(image_input)
        else:
            raise ValueError("不支持的图片输入格式")
        
        # 确保是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # 如果图片太大，进行智能缩放
        if max(original_size) > max_size:
            ratio = max_size / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            
            # 使用高质量的重采样方法
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"图片已缩放: {original_size} -> {new_size}")
        
        return image
    
    def predict_image(self, image_input, save_path=None, show_labels=True, show_conf=True, 
                     conf_threshold=0.25, iou_threshold=0.45, max_size=1920):
        """
        对图片进行推理预测（优化版）
        
        Args:
            image_input: 输入图片
            save_path (str): 保存结果图片的路径
            show_labels (bool): 是否显示标签
            show_conf (bool): 是否显示置信度
            conf_threshold (float): 置信度阈值
            iou_threshold (float): NMS IoU阈值
            max_size (int): 图片最大尺寸限制
            
        Returns:
            PIL.Image: 带有检测结果的图片
        """
        try:
            # 性能计时
            total_start = time.time()
            
            # 1. 预处理
            preprocess_start = time.time()
            image = self.preprocess_image(image_input, max_size)
            preprocess_time = time.time() - preprocess_start
            
            logger.info(f"预处理完成，图片尺寸: {image.size}, 耗时: {preprocess_time:.3f}s")
            
            # 2. 推理
            inference_start = time.time()
            results = self.model(
                image,
                device=self.device,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,  # 固定推理尺寸
                verbose=False,  # 关闭详细输出
                stream=False,   # 不使用流式处理
                save=False,     # 不自动保存
                show=False      # 不显示
            )
            inference_time = time.time() - inference_start
            
            # 3. 后处理
            postprocess_start = time.time()
            
            if results and len(results) > 0:
                result = results[0]
                
                # 使用优化的绘制参数
                annotated_img = result.plot(
                    labels=show_labels,
                    conf=show_conf,
                    line_width=max(1, min(3, image.size[0] // 500)),  # 动态线宽
                    font_size=max(8, min(16, image.size[0] // 100)),  # 动态字体
                    pil=False  # 返回numpy数组，更快
                )
                
                # 高效的颜色空间转换
                if annotated_img.dtype != np.uint8:
                    annotated_img = annotated_img.astype(np.uint8)
                
                # BGR到RGB转换
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(annotated_img_rgb)
                
                # 检测结果统计
                detections = len(result.boxes) if result.boxes is not None else 0
                logger.info(f"检测到 {detections} 个目标")
                
            else:
                # 没有检测到目标，返回原图
                result_image = image
                logger.info("未检测到目标")
            
            postprocess_time = time.time() - postprocess_start
            
            # 4. 保存结果
            save_start = time.time()
            if save_path:
                # 优化保存参数
                result_image.save(save_path, "JPEG", quality=90, optimize=True)
                logger.info(f"结果已保存到: {save_path}")
            save_time = time.time() - save_start
            
            # 总时间统计
            total_time = time.time() - total_start
            
            logger.info(f"性能统计 - 预处理: {preprocess_time:.3f}s, "
                       f"推理: {inference_time:.3f}s, "
                       f"后处理: {postprocess_time:.3f}s, "
                       f"保存: {save_time:.3f}s, "
                       f"总计: {total_time:.3f}s")
            
            return result_image
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise
    
    def predict_batch(self, image_list, output_dir="results", show_labels=True, 
                     show_conf=True, max_workers=4):
        """
        批量推理多张图片（优化版）
        """
        import concurrent.futures
        import threading
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        def process_single_image(args):
            i, image_path = args
            try:
                logger.info(f"处理图片 {i+1}/{len(image_list)}: {image_path}")
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(output_dir, f"detected_{base_name}.jpg")
                
                result_img = self.predict_image(
                    image_path, 
                    save_path=save_path,
                    show_labels=show_labels,
                    show_conf=show_conf
                )
                
                return result_img
                
            except Exception as e:
                logger.error(f"处理图片 {image_path} 失败: {str(e)}")
                return None
        
        # 并行处理（但要注意GPU内存）
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, 2)) as executor:
            futures = [executor.submit(process_single_image, (i, img)) 
                      for i, img in enumerate(image_list)]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return results

# 便捷函数（优化版）
def quick_predict(image_input, model_path="best.engine", save_path=None, device="0"):
    """
    快速推理函数（优化版）
    """
    inference = PCBInference(model_path, device=device)
    return inference.predict_image(image_input, save_path)

# 性能测试函数
def benchmark_model(model_path="best.engine", test_image=None, iterations=10):
    """
    模型性能基准测试
    """
    if test_image is None:
        # 创建测试图片
        test_image = Image.new('RGB', (1920, 1080), color='white')
    
    inference = PCBInference(model_path)
    
    times = []
    for i in range(iterations):
        start = time.time()
        _ = inference.predict_image(test_image)
        end = time.time()
        times.append(end - start)
        print(f"第 {i+1} 次推理: {end - start:.3f}s")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n性能统计:")
    print(f"平均推理时间: {avg_time:.3f}s")
    print(f"FPS: {fps:.1f}")
    print(f"最快: {min(times):.3f}s")
    print(f"最慢: {max(times):.3f}s")

if __name__ == "__main__":
    try:
        # 性能基准测试
        print("开始性能基准测试...")
        benchmark_model("best.engine", iterations=5)
        
        # 实际推理测试
        inference = PCBInference("best.engine")
        
        test_image = "test.jpg"
        if os.path.exists(test_image):
            print(f"\n正在推理图片: {test_image}")
            result = inference.predict_image(
                test_image, 
                save_path="result_optimized.jpg",
                show_labels=True,
                show_conf=True,
                max_size=1920  # 限制最大尺寸
            )
            print("推理完成!")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
