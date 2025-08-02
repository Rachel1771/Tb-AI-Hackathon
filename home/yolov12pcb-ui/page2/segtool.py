import time
import streamlit as st
import requests
import os
from PIL import Image
import io
import tempfile
import sys
import re
from html import unescape
from datetime import datetime

# --- 模块和路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from inference import PCBInference

# --- 页面配置 ---
st.set_page_config(page_title="PCB缺陷检测与分析", page_icon="🔧", layout="wide")

# --- CSS样式 ---
st.markdown("""
<style>
div[data-testid="column"] {
    border: 2px solid #1f77b4 !important;
    border-radius: 15px !important;
    padding: 25px !important;
    margin: 15px 0 !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    min-height: 650px !important;
}
.step-title {
    color: #1f77b4; font-size: 1.5rem; font-weight: bold; margin-bottom: 20px;
    text-align: center; padding: 10px; border-bottom: 2px solid #e0e0e0;
}
.analysis-output-box {
    background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px;
    padding: 20px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    min-height: 350px; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
}
.analysis-waiting {
    display: flex; flex-direction: column; align-items: center; 
    justify-content: center; color: #6c757d; text-align: center; padding: 40px 20px;
}
.analysis-waiting-icon { 
    font-size: 64px; margin-bottom: 20px; opacity: 0.6; 
}
.analysis-waiting-icon.rotating {
    animation: spin 2s linear infinite;
}
.analysis-result { width: 100%; text-align: left; padding: 20px; }
.analysis-result.success { background: #d4edda; border-left: 4px solid #28a745; border-radius: 8px; }
.analysis-result.error { background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 8px; }
.analysis-result.info { background: #d1ecf1; border-left: 4px solid #17a2b8; border-radius: 8px; }
.success-message {
    background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;
    padding: 15px; border-radius: 10px; margin: 10px 0;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulse { 
    0%, 100% { opacity: 0.6; } 
    50% { opacity: 1; } 
}
.pulse { animation: pulse 2s infinite; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 状态初始化 ---
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'detection_time' not in st.session_state:
    st.session_state.detection_time = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# --- 缓存资源 ---
@st.cache_resource
def load_inference_model():
    try:
        return PCBInference("/root/workSpace/tb-hackathon/home/yolov12pcb-ui/page2/data/new-yolov12.engine") 
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

# --- 核心处理函数 ---
def process_detection(uploaded_file, inference_model):
    """处理推理检测，返回结果"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        temp_output_path = tempfile.mktemp(suffix=".jpg")
        start_time = time.time()
        
        result_image = inference_model.predict_image(temp_input_path, save_path=temp_output_path)
        inference_time = time.time() - start_time
        
        if result_image is not None and os.path.exists(temp_output_path):
            with open(temp_output_path, 'rb') as f:
                result_data = f.read()
            
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            return {"success": True, "data": result_data, "time": inference_time}
        else:
            raise Exception("推理失败")
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_analysis(detection_result, dify_api_url, dify_api_key):
    """处理AI分析，返回结果"""
    try:
        image = Image.open(io.BytesIO(detection_result))
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P': image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(img_bytes)
            temp_image_path = tmp_file.name
        
        with open(temp_image_path, 'rb') as f:
            upload_response = requests.post(
                f"{dify_api_url}/files/upload",
                files={'file': ('pcb_analysis.jpg', f, 'image/jpeg')},
                data={'type': 'image'},
                headers={"Authorization": f"Bearer {dify_api_key}"},
                timeout=60
            )
        os.unlink(temp_image_path)
        
        if upload_response.status_code != 201:
            raise Exception(f"文件上传失败: {upload_response.text}")

        file_id = upload_response.json().get('id')
        workflow_payload = {
            "inputs": {"imUrl": {"type": "image", "transfer_method": "local_file", "upload_file_id": file_id}},
            "response_mode": "blocking", "user": "streamlit_user"
        }
        workflow_response = requests.post(
            f"{dify_api_url}/workflows/run",
            json=workflow_payload,
            headers={"Authorization": f"Bearer {dify_api_key}", "Content-Type": "application/json"},
            timeout=120
        )
        
        if workflow_response.status_code == 200:
            result = workflow_response.json()
            analysis_text = result.get("data", {}).get("outputs", {}).get("text", "")
            if analysis_text:
                return {"success": True, "analysis_text": analysis_text, "raw_response": result}
            else:
                raise Exception("工作流执行成功，但没有返回分析文本")
        else:
            raise Exception(f"工作流执行失败: HTTP {workflow_response.status_code}")
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- 页面布局 ---
st.markdown("<div style='text-align: center; padding: 20px;'><h1 style='color: #1f77b4; font-size: 3rem;'>🔧 PCB缺陷检测系统</h1></div>", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 服务配置")
    inference_model = load_inference_model()
    if inference_model: 
        st.success("✅ TensorRT模型已加载")
    dify_api_url = st.text_input("Dify工作流地址", value="https://api.dify.ai/v1")
    dify_api_key = st.text_input("Dify-Api", value="app-YznhSUgv8n9N29bhltjcuXEE", type="password")

# 创建三列布局
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

# --- 第一步：图片上传 ---
with col1:
    st.markdown('<div class="step-title">📤 第一步：上传图片</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("选择PCB图片", type=['png', 'jpg', 'jpeg', 'bmp'], key="file_uploader")
    
    if uploaded_file is not None:
        # 检查是否是新文件，如果是则重置检测和分析结果
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != current_file_id:
            st.session_state.current_file_id = current_file_id
            st.session_state.detection_result = None
            st.session_state.analysis_result = None
            st.session_state.detection_time = None
        
        # 显示上传的图片
        st.image(uploaded_file, caption=f"✅ 已上传: {uploaded_file.name}", use_column_width=True)
        
        # 显示图片信息
        try:
            image = Image.open(uploaded_file)
            st.info(f"📊 尺寸: {image.size[0]}×{image.size[1]} | 格式: {image.format}")
        except Exception as e:
            st.warning(f"无法读取图片信息: {e}")
        
        st.markdown('<div class="success-message">✅ 图片上传成功！</div>', unsafe_allow_html=True)

# --- 第二步：推理检测 ---
with col2:
    st.markdown('<div class="step-title">🔍 第二步：YOLO推理检测</div>', unsafe_allow_html=True)
    
    can_detect = (uploaded_file is not None and 
                  inference_model is not None and 
                  not st.session_state.processing)
    
    # 检测按钮
    if st.button("🚀 开始推理检测", 
                disabled=not can_detect, 
                type="primary", 
                use_container_width=True,
                key="detection_button"):
        
        st.session_state.processing = True
        
        with st.spinner("🔄 正在进行AI推理检测..."):
            result = process_detection(uploaded_file, inference_model)
        
        if result['success']:
            # 保存检测结果
            st.session_state.detection_result = result['data']
            st.session_state.detection_time = result['time']
            st.session_state.analysis_result = None  # 重置分析结果
            
            st.success(f"✅ 推理检测完成！耗时: {result['time']:.2f}秒")
            st.balloons()
        else:
            st.error(f"❌ 推理失败: {result['error']}")
        
        st.session_state.processing = False
        st.rerun()  # 刷新页面以显示结果
    
    # 显示检测结果（只有在不处理时才显示）
    if st.session_state.detection_result is not None and not st.session_state.processing:
        st.subheader("🎯 检测结果")
        st.image(st.session_state.detection_result, caption="缺陷检测结果", use_column_width=True)
        
        # 显示检测时间
        if st.session_state.detection_time:
            st.info(f"⏱️ 检测耗时: {st.session_state.detection_time:.2f}秒")
        
        # 下载按钮
        st.download_button(
            "💾 下载检测结果", 
            st.session_state.detection_result, 
            file_name=f"detected_{uploaded_file.name if uploaded_file else 'result'}.jpg",
            use_container_width=True,
            key="download_detection_result"
        )
        st.markdown('<div class="success-message">✅ 检测完成！可以进行智能分析</div>', unsafe_allow_html=True)

# --- 第三步：智能分析 ---
with col3:
    st.markdown('<div class="step-title">🧠 第三步：智能分析</div>', unsafe_allow_html=True)
    
    can_analyze = (st.session_state.detection_result is not None and 
                   dify_api_key and 
                   not st.session_state.processing and 
                   not st.session_state.analyzing)
    
    # 分析按钮 - 始终在标题下方
    if st.button("🔮 开始智能分析",
                disabled=not can_analyze,
                type="secondary",
                use_container_width=True,
                key="analysis_button"):
        
        st.session_state.analyzing = True
        st.rerun()  # 立即刷新以显示分析状态
    
    # 分析内容区域
    if st.session_state.analyzing:
        # 显示处理中状态（带转圈圈动画）
        st.markdown(f'''
        <div class="analysis-output-box">
            <div class="analysis-waiting">
                <div class="analysis-waiting-icon rotating">🤖</div>
                <div style="font-size: 16px; font-weight: 500; margin-bottom: 10px;">AI正在分析检测结果</div>
                <div style="font-size: 14px; opacity: 0.8;">请稍候，分析过程可能需要1-2分钟...</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # 执行分析
        result = process_analysis(
            st.session_state.detection_result,
            dify_api_url,
            dify_api_key
        )
        
        # 保存分析结果并结束分析状态
        st.session_state.analysis_result = result
        st.session_state.analyzing = False
        
        if result['success']:
            st.success("✅ 智能分析完成！")
        else:
            st.error(f"❌ 分析失败: {result['error']}")
        
        st.rerun()  # 刷新以显示最终结果
    
    elif st.session_state.analysis_result:
        # 显示分析结果
        result = st.session_state.analysis_result
        
        if result['success']:
            analysis_text = result['analysis_text']
            
            # 判断结果类型
            if any(keyword in analysis_text for keyword in ["缺陷", "缺失", "问题", "错误", "故障", "异常"]):
                result_class, icon, title = "analysis-result error", "⚠️", "发现缺陷"
            elif any(keyword in analysis_text for keyword in ["正常", "良好", "无问题", "合格", "完好"]):
                result_class, icon, title = "analysis-result success", "✅", "检测通过"
            else:
                result_class, icon, title = "analysis-result info", "📋", "检测结果"
            
            st.markdown(f'''
            <div class="analysis-output-box">
                <div class="{result_class}">
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 12px;">{icon} {title}</div>
                    <div style="font-size: 14px; line-height: 1.6;">{analysis_text}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 导出报告按钮
            clean_text = re.sub(r'<[^>]*>', '', unescape(analysis_text))
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text.strip())
            
            report = f"""PCB检测分析报告
================
分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
检测结果：{clean_text}

详细信息：
- 图片格式：JPEG
- 分析引擎：Dify AI工作流
- 检测类型：PCB缺陷检测
"""
            
            st.download_button(
                label="📄 导出报告",
                data=report,
                file_name=f"pcb_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="export_report"
            )
        else:
            st.markdown(f'''
            <div class="analysis-output-box">
                <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">❌ 分析失败</div>
                    <div style="font-size: 14px;">{result['error']}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    else:
        # 显示等待状态
        if not can_analyze:
            if not dify_api_key:
                icon, main_text, sub_text, icon_class = "🔑", "请先配置Dify API Key", "在左侧边栏中输入有效的API密钥", ""
            else:
                icon, main_text, sub_text, icon_class = "🔍", "请先完成推理检测", "上传图片并运行YOLO检测后即可分析", ""
        else:
            icon, main_text, sub_text, icon_class = "🔄", "准备就绪", "点击按钮开始智能分析", "pulse"
        
        st.markdown(f'''
        <div class="analysis-output-box">
            <div class="analysis-waiting">
                <div class="analysis-waiting-icon {icon_class}">{icon}</div>
                <div style="font-size: 16px; font-weight: 500; margin-bottom: 10px;">{main_text}</div>
                <div style="font-size: 14px; opacity: 0.8;">{sub_text}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
