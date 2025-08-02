import streamlit as st
from st_pages import add_page_title, hide_pages
from PIL import Image
add_page_title(layout="wide")

hide_pages(["Thank you"])

st.success("""
* [国家健康数据中心](https://www.ncmi.cn/)
* [肝脏肿瘤数据中心](http://www.nltds.org/)
""")

st.markdown("""
欢迎来到核心组件介绍页，本模型基于Unet网络架构进行创新，提出了多尺度特征融合和双维度注意力的Unet网络结构Multi Scale Feature Extraction and Two-Dimensional Attention Unet(MTDUnet)
""")

st.markdown("---")

# st.markdown("""##  别走开哦，有惊喜，如果你喜欢这个页面，请给作者来一颗小星星⭐⭐！""", unsafe_allow_html=True)
#
# st.markdown("---")

st.markdown("## 1️⃣MTDUnet整体设计")
st.markdown("结构上保留了跳跃连接，使得模型为U型结构。编码器部分首先会对输入的3通道图像进行处理，经过1x1卷积扩大通道数。"
            "MSFF模块是由空间金字塔特征提取和CA注意力机制组合形成的残差结构，空间金字塔由膨胀率为1、2、4的3x3膨胀卷积、全局自适应平均池化和BatchNormal构成。"
            "提取和融合特征的同时不会改变特征的尺寸，下采样块采用经典的两个3x3卷积和2x2最大池化组成，浓缩特征，降低分辨率。"
            "解码器加入了RRLA模块，由3x3的深度可分离卷积和膨胀率为1的3x3空洞卷积构成主干，CBAM注意力经Residual连接构成残差结构，旨在更好的利用全局语义特征，"
            "上采样使用反卷积法，两个3x3的反卷积和Relu组成。最后经过一个1x1的卷积恢复通道数后形成图片输出。")
MTDUnet = st.container()
image1 = Image.open('page2/image/block/MTDUnet.png')
# image_container1.header("膈面")
MTDUnet.image(image1, caption='MTDUnet网络架构', use_column_width=True, output_format='PNG')




st.markdown("---")

st.markdown("## 2️⃣多尺度特征融合模块")
st.markdown("该模块是借鉴ResNet结构改进的。在网络浅层部分进行特征提取的过程中，"
            "需要更大的感受野和捕获全局语义信息，设计使用空洞空间金字塔提取模块ASPM。"
            "采用三个3x3膨胀率为[1,2,4]的空洞卷积进行多尺度特征的提取，得益于空段卷积的特性，"
            "使得在浅层网络拥有了更大的感受野，对三个尺度的特征进行拼接后经过1x1卷积核恢复通道数。"
            "使用全局平均池化替代了最大池化，在浅层网络的初步提取过程中使用最大池化用以损失一些局部纹理信息。"
            "经过1x1的卷积核融合，最后拼接进行输出。分别使用了relu进行激活和BatchNormal。"
            "经过Concat后的多尺度特征会输入到CA注意力机制模块中进行特征融合，通过对不同维度上信息建模，自适应调整特征权重，"
            "在最后会进行残差连接后输出。这个设计在低层级网络中较好的融合了局部和全局的信息，把握浅层网络的局部细节和纹理并跳跃连接到解码器中，"
            "与高层级语义一起共同进行解码。得益于ResNet残差连接，将该模块插入到上采样之前，可以保证效果不会低于原始的Unet编码结构。")
MSFF = st.container()
image2 = Image.open('page2/image/block/MSFF.png')
# image_container1.header("膈面")
MSFF.image(image2, caption='多尺度特征融合模块', use_column_width=True, output_format='PNG')


st.markdown("---")

st.markdown("## 3️⃣残差回型锁注意力机制")
st.markdown("RRLA插入在Decoder部分中上采样模块的后面。RRLA模块的设计借鉴ResNet残差网络设计，"
            "主分支由一个3x3膨胀率为1的空洞卷积搭配一个3x3的深度可分离卷积构成。残差分支由一个CBAM注意力机制模块构成，"
            "相加后送入Relu和BatchNormal。在RRLA模块外部再加入残差连接，构成一个回型锁的结构。"
           " 经2.1.4节介绍，上采样层的主要作用就是恢复图像分辨率，但是会损失部分精度和特征信息，"
            "为了充分运用浅层网络跳跃连接的低层级语义特征，使用膨胀卷积搭配深度可分离卷积进行处理，扩大感受野的同时缓解拟合，"
            "并减少了解码器部分的计算量。为更好的利用全局语义信息，将CBAM模块插入到残差结构中，在通道和空间两个维度上使用注意力机制来引导网络进行自学习，"
            "兼顾低层级语义中的局部纹理和全局语义信息，更有效地关注显著性特征。得益于ResNet残差连接，可以保持并超越原始卷积运算的性能，提高了模型的鲁棒性。")
RRLA = st.container()
image3 = Image.open('page2/image/block/RRLA.png')
# image_container1.header("膈面")
RRLA.image(image3, caption='残差回型锁注意力机制', use_column_width=True, output_format='PNG')




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
