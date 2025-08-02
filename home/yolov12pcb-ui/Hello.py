import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title, hide_pages

add_page_title()

show_pages(
    [
        Page("Hello.py", "欢迎来到Rachel的分割工具箱", "💻"),

        # 菜单一
        # Section("友爱之家", "🧙‍♂️"),
        Section("友爱之家", "👨‍⚕️️"),
        Page("page1/introduction.py", "小小科普", "📚", in_section=True),
        Page("page1/tools.py", "技术指南", "🛠️", in_section=True),

        # 菜单二
        Section("工具仓库", "🧑‍💻"),
        # Page("page2/block.py", "核心组件", "📇", in_section=True),
        Page("page2/segtool.py", "分割工具", "🗃️", in_section=True),
        Page("page2/eva.py", "数据指标", "📊", in_section=True),
    ]
)

hide_pages(["Thank you"])

st.markdown("### 👨‍🔧 感谢[DataTalksClub](https://datatalks.club/)")

st.image("https://pbs.twimg.com/media/FmmYA2YWYAApPRB.png")

# st.info("Original Course Repository on [Github](https://github.com/DataTalksClub/data-engineering-zoomcamp)")

st.markdown("---")

# with st.expander("Sign up here for 2024 Cohort"):
#     st.markdown("""
#
#     <a href="https://airtable.com/appzbS8Pkg9PL254a/shr6oVXeQvSI5HuWD"><img src="https://user-images.githubusercontent.com/875246/185755203-17945fd1-6b64-46f2-8377-1011dcb1a444.png" height="50" /></a>
#
#     #
#
#     - Register in [DataTalks.Club's Slack](https://datatalks.club/slack.html)
#     - Join the [`#course-data-engineering`](https://app.slack.com/client/T01ATQK62F8/C01FABYF2RG) channel
#     - Join the [course Telegram channel with announcements](https://t.me/dezoomcamp)
#     - The videos are published on [DataTalks.Club's YouTube channel](https://www.youtube.com/c/DataTalksClub) in [the course playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb)
#     - [Frequently asked technical questions](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit?usp=sharing)
#
#     #""", unsafe_allow_html=True)

st.markdown("""
### 👨‍🎓 工具箱说明

##### 👨‍⚕️️友爱之家

* 小小科普：对肝脏进行科普简介
* 技术指南：概要介绍本项目使用的技术工具



##### 🧑‍💻 工具仓库

* 分割工具：项目核心组件，提供肝脏肿瘤图像分割工具
* 数据指标：本项目参与比较的模型训练各指标对比情况

### 🔎 Overview""", unsafe_allow_html=True)

st.image(
    "https://raw.githubusercontent.com/DataTalksClub/data-engineering-zoomcamp/main/images/architecture/photo1700757552.jpeg")

# st.markdown("""
# ### 📓 Prerequisites
#
# To get the most out of this course, you should feel comfortable with coding and command line
# and know the basics of SQL. Prior experience with Python will be helpful, but you can pick
# Python relatively fast if you have experience with other programming languages.
#
# Prior experience with data engineering is not required.
#
# ### 👨‍🏫 Instructors
#
# - [Ankush Khanna](https://linkedin.com/in/ankushkhanna2)
# - [Victoria Perez Mola](https://www.linkedin.com/in/victoriaperezmola/)
# - [Alexey Grigorev](https://linkedin.com/in/agrigorev)
# - [Matt Palmer](https://www.linkedin.com/in/matt-palmer/)
# - [Luis Oliveira](https://www.linkedin.com/in/lgsoliveira/)
# - [Michael Shoemaker](https://www.linkedin.com/in/michaelshoemaker1/)
#
# Past instructors:
#
# - [Sejal Vaidya](https://www.linkedin.com/in/vaidyasejal/)
# - [Irem Erturk](https://www.linkedin.com/in/iremerturk/)
#
# ### ❔ Asking for help in Slack
#
# The best way to get support is to use [DataTalks.Club's Slack](https://datatalks.club/slack.html). Join the [`#course-data-engineering`](https://app.slack.com/client/T01ATQK62F8/C01FABYF2RG) channel.
#
# To make discussions in Slack more organized:
#
# * Follow [these recommendations](asking-questions.md) when asking for help
# * Read the [DataTalks.Club community guidelines](https://datatalks.club/slack/guidelines.html)
# """)
st.markdown("""
---

##### 🤪 界面设计来自[2000300309陈乐](https://github.com/Rachel1771)
""", unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)