import streamlit as st
from st_pages import add_page_title, hide_pages
from PIL import Image
add_page_title(layout="wide")

hide_pages(["Thank you"])

st.success("""
* 深度学习
* 计算机视觉
* 神经网络
* 医学图像分割
""")

st.markdown("""
欢迎来到技术指南之家，在这里将会为您解答作者是如何进行肝脏肿瘤分割任务的
""")

st.markdown("---")

st.markdown("""##  在这里，可以从零开始了解肝脏肿瘤分割任务🛩️""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 1️⃣视觉任务👁️")
st.markdown("计算机视觉任务通俗来讲就是运用计算机的高性能处理能力来模拟人脑处理图像的过程，我们肉眼看到一只狗，能够识别出来是狗，因为狗有狗的特征，有他的颜色、形状等等特征，通过RGB"
            "三色映射到我们的视觉中，大脑处理让我们分辨出来这是狗。但是一张图片在计算机上的表示就是一堆数据，这些数据也是可以进行学习和处理的，那我们可以假设设计一个“类大脑的处理程序”"
            "输入一张图片（当然，表示起来都是数字），经过这个程序，输出一个辨别的结果，这就是一个简单的分类任务。")
image = Image.open('page1/image/head.png')
st.image(image,caption='图片数据', use_column_width=True, output_format='PNG')
st.markdown("可以发现上面的一张马赛克黑白图，是可以表示成一堆数据，我们可以用列表、数组、或者矩阵这样的方式来存储它，你是知道的，计算机处理数字的能力是当然强于我们的，所以视觉任务的"
            "核心就落点在设计一个“类大脑的处理程序”")

st.markdown("### 深度学习🕵️‍♂️")
st.markdown("随着近年来深度学习的高速发展，现在的计算机视觉任务基本都是使用深度学习来完成的，深度学习方法不需要人工，而是依赖算法自动提取特征。"
            "深度学习模仿人类大脑的运行方式，从经验中学习获取知识。这也是深度学习被看做黑盒子，可解释性差的原因。"
            "**神经网络**是常用于解决计算机视觉任务的方法，神经网络是模拟人脑神经元的工作工程"
            "由许多“神经元”构成，这些神经元有着自己的功能，能够前向和反向的传播给网络，及时的调整整体的学习信息"
            "我们的任务就是搭建一个合理的神经网络模型，能够根据一些给定的训练数据集，学习数据集的信息特征，将学习到的参数"
            "用于预测中。我们可以从下图大致了解视觉任务和神经网络的关系。")
image = Image.open('page1/image/deeplearning.png')
st.image(image,caption='深度学习', use_column_width=True, output_format='PNG')
image = Image.open('page1/image/relation.png')
st.image(image,caption='深度学习一般方法', use_column_width=True, output_format='PNG')
st.markdown("---")
st.markdown("##  2️⃣肝脏肿瘤分割🧮")
st.success("""
医学图像分割的一个难点在于处理数据，相关的模型大体上差别不多。我们平时对正常图片处理的时候格式都是JPG或者PNG的格式，但是肝脏CT
是nil格式的，不是正常图片，而且CT是对腹部进行扫描，会有很多其他的器官组织以及脊柱的干扰。数据预处理是一个非常重要的环节，我们需要将
CT进行切片，并且进行处理，将肝脏部分更好的展示出来，常用的方法一般都是去噪、归一化、数据增强、随机反转、弹性伸缩等。
""")
# image = Image.open('page1/image/ct1.png')
# st.image(image,caption='CT处理后', use_column_width=True, output_format='PNG')

st.markdown("### 数据集💾")
st.success("""
LiTS (The Liver Tumor Segmentation Benchmark) 是专注于肝脏及其肿瘤分割的 CT 数据集。
该数据集收集了 7 个不同医学中心的数据，包含 131 例训练集和 70 例测试集。
基于该数据集，已在 ISBI 2017，MICCAI 2017 和 MICCAI 2018 都成功举办了相关竞赛，
并被 MSD (Medical Segmentation Decathlon) 收作 Task03 子任务。
[🔍LITS🔍](https://paperswithcode.com/dataset/lits17)
""")
st.markdown("---")
st.markdown("## 3️⃣学习框架🪜")
st.markdown("有了数据集，并且进行了处理，我们就该进行模型的设计来训练数据了。神经网络是相当复杂的，如果单纯使用"
            "编程语言来实现其中的数学逻辑，但是一个卷积块就会占一定的工作量，更不用说各种复杂的深层次网络设计，"
            "所以学者们设计出来了深度学习框架，集成了很多基础函数和模块，我们只需要更专注于网络的设计即可。目前主流的深度学习"
            "框架有很多"
            "[PyTorch](https://pytorch.org)、"
            "[TensorFlow](https://www.tensorflow.org)、"
            "PaddlepPaddle(作者推荐😚)](https://www.paddlepaddle.org.cn/)")
st.markdown("### PaddlePaddle😍")
st.success("""
*  国产深度学习框架
*  业内首个动静统一的框架，动态图编程调试转静态图预测部署
* 业内首个通用异构参数服务器架构，端到端自适应分布式训练架构
* 即训即用，支持端边云多硬件和多操作系统
* 算法总数超过600个，包含领先的预训练模型
""")
image = Image.open('page1/image/paddle.png')
st.image(image,caption='飞浆组件', use_column_width=True, output_format='PNG')

st.markdown("---")
st.markdown("## 4️⃣骨架网络🦴")
st.success("""
    一般我们在进行视觉等深度学习任务时都会使用一些骨架网络，这些网络往往在相对应任务的领域会有着较好的一个性能表现，
    我们需要再骨架网络的基础上进行个人创新和改进，以提高网络的性能。本次肝脏肿瘤分割中，作者进行的是2D平面分割，在
    医学图像领域用的比较多的是Ronneberge于2015年提出的UNet网络模型，这是一个非常经典的网络模型，能够在较小的数据集上
    获得较优的性能。
    [😊UNet论文出处😊](https://arxiv.org/abs/1505.04597)
""")
st.markdown("UNet由一个编码器和解码器构成，解码器。编码器主要由一组卷积块构成，不断进行下采样以获得丰富的特征信息，下采样过程"
            "中通道数会不断减少，尺寸也变小，但是语义信息逐渐丰富；解码器由一组反卷积构成，上采样过程中恢复图片尺寸，并对"
            "丰富信息进行表征；这个U型的来源是在下采样过程中产生的特征会拼接到上采样中，所以把网络给掰弯了成了个U型")
image = Image.open('page1/image/UNet.png')
st.image(image,caption='UNet网络', use_column_width=True, output_format='PNG')

st.markdown("## 5️⃣后续处理🪚")
st.markdown("这里只是带大家简单了解一些任务的过程，所以不会详细介绍模型设计、训练和预测等细节。我们在上一节中选好了骨架网络"
            "并设置好参数进行训练后，会得到一个权重参数文件，这就是我们要的模型，这一组参数是神经网络不断学习出来的最优解。"
            "我们可以直接在深度学习框架中调用API进行预测，以评估其结果。如果需要模型迁移，这是非常重要的一点，因为不同学习框架"
            "的权重参数文件格式不一，所以要兼容十分困难，我们需要一种统一的格式进行部署，要尽可能简洁、不需要高性能、推理速度也要快，"
            "这时候ONNX就来啦！")
st.markdown("### ONNX🗑️")
st.markdown("简单描述一下官方介绍，开放神经网络交换（Open Neural Network Exchange）简称ONNX是微软和Facebook提出用来表示深度学习模型的开放格式。"
            "所谓开放就是ONNX定义了一组和环境，平台均无关的标准格式，来增强各种AI模型的可交互性。"
            "换句话说，无论你使用何种训练框架训练模型（比如TensorFlow/Pytorch/OneFlow/Paddle），在训练完毕后你都可以将这些框架的模型统一转换为ONNX这种统一的格式进行存储。"
            "注意ONNX文件不仅仅存储了神经网络模型的权重，同时也存储了模型的结构信息以及网络中每一层的输入输出和一些其它的辅助信息。"
            "在本地进行推理不需要GPU资源，而且速度很快，部署到各个平台都可以进行使用。")

st.success("""
*  训练好模型
*  转为ONNX
* 各个平台部署
* 调用模型进行开发
""")

st.markdown("---")
st.markdown("## 6️⃣学习推荐🤡")
st.markdown("### 深度学习入门🔥")
st.video("https://www.youtube.com/watch?v=rDhBcP4ikpA&list=PLDzdzeKX7DWep2KyJwJ-BmYciXTARvZO6&index=1")
st.markdown("### 卷积神经网络🔥")
st.video("https://www.youtube.com/watch?v=Z5qJ9IxSuKo")
st.markdown("### 计算机视觉🔥")
st.video("https://www.youtube.com/watch?v=_lKfP-N2Z3A&list=PL7kmjdSJ3kmARWiXOhhl12PqyHe4jcjxv")
st.markdown("""
---

## Community notes

* [Notes from Alvaro Navas](https://github.com/ziritrion/dataeng-zoomcamp/blob/main/notes/1_intro.md)
* [Notes from Abd](https://itnadigital.notion.site/Week-1-Introduction-f18de7e69eb4453594175d0b1334b2f4)
* [Notes from Aaron](https://github.com/ABZ-Aaron/DataEngineerZoomCamp/blob/master/week_1_basics_n_setup/README.md)
* [Notes from Faisal](https://github.com/FaisalMohd/data-engineering-zoomcamp/blob/main/week_1_basics_n_setup/Notes/DE%20Zoomcamp%20Week-1.pdf)
* [Michael Harty's Notes](https://github.com/mharty3/data_engineering_zoomcamp_2022/tree/main/week01)
* [Blog post from Isaac Kargar](https://kargarisaac.github.io/blog/data%20engineering/jupyter/2022/01/18/data-engineering-w1.html)
* [Handwritten Notes By Mahmoud Zaher](https://github.com/zaherweb/DataEngineering/blob/master/week%201.pdf)
* [Notes from Candace Williams](https://teacherc.github.io/data-engineering/2023/01/18/zoomcamp1.html)
* [Notes from Marcos Torregrosa](https://www.n4gash.com/2023/data-engineering-zoomcamp-semana-1/)
* [Notes from Vincenzo Galante](https://binchentso.notion.site/Data-Talks-Club-Data-Engineering-Zoomcamp-8699af8e7ff94ec49e6f9bdec8eb69fd)
* [Notes from Victor Padilha](https://github.com/padilha/de-zoomcamp/tree/master/week1)
* [Notes from froukje](https://github.com/froukje/de-zoomcamp/blob/main/week_1_basics_n_setup/notes/notes_week_01.md)
* [Notes from adamiaonr](https://github.com/adamiaonr/data-engineering-zoomcamp/blob/main/week_1_basics_n_setup/2_docker_sql/NOTES.md)
* [Notes from Xia He-Bleinagel](https://xiahe-bleinagel.com/2023/01/week-1-data-engineering-zoomcamp-notes/)
* [Notes from Balaji](https://github.com/Balajirvp/DE-Zoomcamp/blob/main/Week%201/Detailed%20Week%201%20Notes.ipynb)
* [Notes from Erik](https://twitter.com/ehub96/status/1621351266281730049)
* [Notes by Alain Boisvert](https://github.com/boisalai/de-zoomcamp-2023/blob/main/week1.md)
* Notes on [Docker, Docker Compose, and setting up a proper Python environment](https://medium.com/@verazabeida/zoomcamp-2023-week-1-f4f94cb360ae), by Vera
* [Setting up the development environment on Google Virtual Machine](https://itsadityagupta.hashnode.dev/setting-up-the-development-environment-on-google-virtual-machine), blog post by Aditya Gupta 

---

##### 🤪 界面设计来自[2000300309陈乐](https://github.com/Rachel1771)
""")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
