import time
import streamlit as st
import requests
import os
from PIL import Image
import io
import base64

# 页面配置
st.set_page_config(
    page_title="PCB缺陷检测与分析",
    page_icon="🔧",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
.step-container {
    border: 2px solid #e1e5e9;
    border-radius: 15px;
    padding: 25px;
    margin: 15px 0;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.step-header {
    color: #1f77b4;
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
}

.upload-area {
    border: 3px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: #fafafa;
}

.success-message {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

.error-message {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# 主标题
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #1f77b4; font-size: 3rem;'>🔧 PCB缺陷检测系统</h1>
    <p style='color: #666; font-size: 1.2rem;'>上传图片 → AI推理 → 智能分析</p>
</div>
""", unsafe_allow_html=True)

# API配置（简化版，放在侧边栏）
with st.sidebar:
    st.header("⚙️ 服务配置")
    gpu_server_url = st.text_input("GPU推理服务", value="http://localhost:8000")
    dify_api_url = st.text_input("Dify工作流地址", value="http://localhost:3000")
    dify_api_key = st.text_input("Dify API Key", type="password")

# 主要工作流程
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

# 第一步：图片上传
with col1:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">📤 第一步：上传图片</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "选择PCB图片",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=False,
        key="pcb_upload"
    )

    if uploaded_file is not None:
        # 保存上传的图片到session state
        st.session_state.uploaded_image = uploaded_file

        # 显示上传的图片
        st.image(uploaded_file, caption=f"✅ 已上传: {uploaded_file.name}", use_column_width=True)

        # 图片信息
        image = Image.open(uploaded_file)
        st.info(f"📊 尺寸: {image.size[0]}×{image.size[1]} | 格式: {image.format}")

        st.markdown('<div class="success-message">✅ 图片上传成功！可以进行推理检测</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-area">
            <h3>📁 拖拽或点击上传PCB图片</h3>
            <p>支持 PNG, JPG, JPEG, BMP 格式</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 第二步：AI推理检测
with col2:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">🔍 第二步：AI推理检测</div>', unsafe_allow_html=True)

    # 推理按钮
    inference_disabled = st.session_state.uploaded_image is None

    if st.button(
            "🚀 开始推理检测",
            disabled=inference_disabled,
            type="primary",
            use_container_width=True
    ):
        if st.session_state.uploaded_image is not None:
            with st.spinner("🔄 正在进行AI推理检测..."):
                try:
                    # 调用GPU推理API
                    files = {
                        "file": (
                            st.session_state.uploaded_image.name,
                            st.session_state.uploaded_image.getvalue(),
                            st.session_state.uploaded_image.type
                        )
                    }

                    response = requests.post(
                        f"{gpu_server_url}/api/pcb-detect",
                        files=files,
                        timeout=60
                    )

                    if response.status_code == 200:
                        # 保存检测结果
                        st.session_state.detection_result = response.content
                        st.success("✅ 推理检测完成！")
                        st.balloons()  # 庆祝动画
                    else:
                        st.error(f"❌ 推理失败: HTTP {response.status_code}")
                        st.session_state.detection_result = None

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ 连接失败: {str(e)}")
                    st.session_state.detection_result = None

    # 显示推理结果
    if st.session_state.detection_result is not None:
        st.subheader("🎯 检测结果")
        result_image = Image.open(io.BytesIO(st.session_state.detection_result))
        st.image(result_image, caption="缺陷检测结果", use_column_width=True)

        # 下载按钮
        st.download_button(
            label="💾 下载检测结果",
            data=st.session_state.detection_result,
            file_name=f"detected_{st.session_state.uploaded_image.name}" if st.session_state.uploaded_image else "detection_result.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

        st.markdown('<div class="success-message">✅ 检测完成！可以进行智能分析</div>', unsafe_allow_html=True)
    else:
        if inference_disabled:
            st.info("⚠️ 请先上传图片")
        else:
            st.info("🔄 等待推理结果...")

        # 占位图片
        placeholder_img = Image.new('RGB', (400, 300), (240, 240, 240))
        st.image(placeholder_img, caption="等待检测结果", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 第三步：智能分析
with col3:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">🧠 第三步：智能分析</div>', unsafe_allow_html=True)

    # 分析按钮
    analysis_disabled = st.session_state.detection_result is None or not dify_api_key

    if st.button(
            "🔮 开始智能分析",
            disabled=analysis_disabled,
            type="secondary",
            use_container_width=True
    ):
        if st.session_state.detection_result is not None and dify_api_key:
            with st.spinner("🤖 AI正在分析检测结果..."):
                try:
                    # 将检测结果图片转换为base64
                    img_base64 = base64.b64encode(st.session_state.detection_result).decode()

                    # 调用Dify工作流
                    dify_payload = {
                        "inputs": {
                            "image": img_base64,
                            "filename": st.session_state.uploaded_image.name if st.session_state.uploaded_image else "unknown.jpg"
                        },
                        "response_mode": "blocking",
                        "user": "streamlit_user"
                    }

                    headers = {
                        "Authorization": f"Bearer {dify_api_key}",
                        "Content-Type": "application/json"
                    }

                    response = requests.post(
                        f"{dify_api_url}/v1/workflows/run",
                        json=dify_payload,
                        headers=headers,
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.analysis_result = result.get("data", {}).get("outputs", {})
                        st.success("✅ 智能分析完成！")
                        st.balloons()  # 庆祝动画
                    else:
                        st.error(f"❌ 分析失败: HTTP {response.status_code}")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Dify连接失败: {str(e)}")

    # 显示分析结果
    if st.session_state.analysis_result:
        st.subheader("📊 分析报告")

        analysis = st.session_state.analysis_result

        # 根据你的Dify工作流输出调整这里
        if "summary" in analysis:
            st.markdown(f"**🔍 缺陷总结:**")
            st.write(analysis["summary"])

        if "defect_count" in analysis:
            st.metric("发现缺陷数量", analysis["defect_count"])

        if "severity" in analysis:
            severity = analysis["severity"]
            if severity.lower() in ["高", "严重", "high"]:
                st.error(f"⚠️ 严重程度: {severity}")
            elif severity.lower() in ["中", "中等", "medium"]:
                st.warning(f"⚠️ 严重程度: {severity}")
            else:
                st.success(f"✅ 严重程度: {severity}")

        if "recommendations" in analysis:
            st.markdown(f"**💡 处理建议:**")
            st.write(analysis["recommendations"])

        if "confidence" in analysis:
            confidence = float(analysis["confidence"])
            st.progress(confidence / 100)
            st.caption(f"分析可信度: {confidence:.1f}%")

    else:
        if analysis_disabled:
            if not dify_api_key:
                st.warning("⚠️ 请先配置Dify API Key")
            else:
                st.info("⚠️ 请先完成推理检测")
        else:
            st.info("🔄 等待分析结果...")

    st.markdown('</div>', unsafe_allow_html=True)

# 底部操作区域
st.markdown("---")
col_reset, col_status = st.columns([1, 3])

with col_reset:
    if st.button("🔄 重置所有步骤", use_container_width=True):
        st.session_state.uploaded_image = None
        st.session_state.detection_result = None
        st.session_state.analysis_result = None
        st.rerun()

with col_status:
    # 显示当前进度
    progress_steps = []
    if st.session_state.uploaded_image is not None:
        progress_steps.append("✅ 图片已上传")
    if st.session_state.detection_result is not None:
        progress_steps.append("✅ 推理已完成")
    if st.session_state.analysis_result is not None:
        progress_steps.append("✅ 分析已完成")

    if progress_steps:
        st.success(" | ".join(progress_steps))
    else:
        st.info("📋 请开始第一步：上传PCB图片")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🔧 PCB缺陷检测与智能分析系统</p>
    <p>YOLOv12 + TensorRT + Dify工作流 | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
