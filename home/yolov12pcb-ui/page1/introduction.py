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
欢迎您来到肝脏安全健康关心之家，通过上面两个网站我们可以了解到跟肝脏疾病相关等数据，我们将在下面以简短的文字介绍跟肝脏相关的知识。
""")

st.markdown("---")

st.markdown("""##  别走开哦，有惊喜，如果你喜欢这个页面，请给作者来一颗小星星⭐⭐！""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 1️⃣五脏之一")
st.markdown("肝（liver），人体脏器名，五脏之一。是脊椎动物身体内以代谢功能为主的一个器官，并在身体里面充分扮演着去氧化，储存肝糖，分泌性蛋白质的合成等等。肝脏也制造消化系统中之胆汁。在医学用字上，常以拉丁语字首hepato-或hepatic来描述肝脏。大部分的肝脏疾病都会有黄疸的症状，这是由于肝脏无法继续将胆红素排出所以就在体内累积。中医认为：肝与胆相为表里，开窍于目，肝主藏血，主疏泄，有贮藏和调节血液的功能。《素问·五脏生成》：“肝之合筋也，其荣爪也。”肝又为将军之官，主谋虑。")

st.markdown("---")
st.markdown("## 2️⃣肝脏的功能🎥")
st.video("https://www.youtube.com/watch?v=T3YTWpKskRY")

st.markdown("""
  * 代谢方面：维生素、激素、抗利尿激素
  * 分泌和排泄胆汁
  * 解毒
  * 胎儿时期是重要的造血器官，维持人体凝血和抗凝动态平衡
  """)

st.markdown("---")
st.markdown("## 3️⃣现状📝")
st.markdown(" 肝脏是腹腔内最大的实质性器官，具有代谢、解毒、免疫防御等生理功能。肝脏肿瘤是发生在肝脏的异常组织增生，可分为肝脏良性肿瘤和肝脏恶性肿瘤(也称为肝癌)据统计,我国肝癌发病率占所有癌症的9%，居癌症发病谱第4位；肝癌死亡率占所有癌症的14%居癌症死亡谱第2位；通过查阅文献，10966例肝癌患者的临床资料进行回顾性分析，发现早期肝癌患者经治疗，其5年生存率高达62.5%~77.2%，而中、晚期肝癌患者的5年生存率仅23.8%因此，尽早对肝脏肿瘤进行良恶性诊断对于及时诊断患者肝脏情况维护肝脏健康提高肝癌生存率具有重要意义。")

st.success("""
* 所以，肝脏如此重要，我们应该如何在平时生活中保护它呢？
""")
st.markdown("####  自检🤏")
st.video("https://www.youtube.com/watch?v=xZ-oMOXXbQw")
st.markdown("####  饮食🍝")
st.video("https://www.youtube.com/watch?v=X9KXX5lI2CU")
st.markdown("#### 运动🏃‍♂️")
st.video("https://www.youtube.com/watch?v=Zj5DQ-t1Ago")
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
