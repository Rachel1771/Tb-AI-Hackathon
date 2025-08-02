import streamlit as st
from st_pages import add_page_title, hide_pages
from PIL import Image
add_page_title(layout="wide")

hide_pages(["Thank you"])
st.balloons()
st.success("""
感谢这些工具箱帮我实现了指标数据的可视化和模型部署：
* [📀ONNX](https://onnx.ai/)
* [📀PaddlePaddle](https://www.paddlepaddle.org.cn/)
* [📀Netron](https://netron.app/)
""")
st.markdown("此页面用来呈现模型的评估指标和训练参数在训练过程中的变化")
st.markdown("---")
st.markdown("## 本次的是🎖MTDUnet!!!")
cols = st.columns(2)


def st_code_block(url, caption=None, code=None):
    # prefill the http address for the sql-reference url
    if not url.startswith("https"):
        url = f"https://docs.snowflake.com/en/sql-reference/sql/{url}"
    st.caption(f"[☁️]({url}) {caption}")
    st.code(code, language="sql")


def Unet():
    st.header("1️⃣ Unet", help="No")
    mIoU, Acc, mRecall, batch_cost, train_loss, reader_cost = \
        st.tabs(["mIoU", "Acc", "mRecall",  "batch_cost", "train_loss","reader_cost"])
    with mIoU:
        st.image(Image.open("page2/image/unet/Evaluate_mIoU.png"))
    with Acc:
        st.image(Image.open("page2/image/unet/Evaluate_Acc.png"))
    with mRecall:
        st.image(Image.open("page2/image/unet/Evaluate_mRecall.png"))
    with batch_cost:
        st.image(Image.open("page2/image/unet/Train_batch_cost.png"))
    with train_loss:
        st.image(Image.open("page2/image/unet/Train_loss.png"))
    with reader_cost:
        st.image(Image.open("page2/image/unet/Train_reader_cost.png"))

    # with tips_tab:
    #     st.markdown("""
    #     💡 **Tips**
    #     - Follow consistent and meaningful naming conventions for `DATABASE` objects.
    #     - When you create a new Snowflake database, it also generates two schemas: `PUBLIC` (the default schema) and `INFORMATION_SCHEMA` (containing views and table functions for querying metadata across objects).
    #     - Use a `TRANSIENT` `DATABASE` to isolate temporary data, and provide a dedicated space for intermediate results or temporary tables during specific analysis or transformation tasks.
    #     - Utilize zero-copy cloning using `CREATE DATABASE <name> CLONE <source_db>` for efficient, space-saving `DATABASE` copies.
    #     - Continuously analyze query and resource usage patterns to fine-tune `DATABASE` parameters for optimal performance and cost efficiency.
    #     """
    #                 )
def Unetpp():
    st.header("2️⃣ Unet++", help="No")
    mIoU, Acc, mRecall, batch_cost, train_loss, reader_cost = \
        st.tabs(["mIoU", "Acc", "mRecall",  "batch_cost", "train_loss","reader_cost"])
    with mIoU:
        st.image(Image.open("page2/image/unet++/Evaluate_mIoU.png"))
    with Acc:
        st.image(Image.open("page2/image/unet++/Evaluate_Acc.png"))
    with mRecall:
        st.image(Image.open("page2/image/unet++/Evaluate_mRecall.png"))
    with batch_cost:
        st.image(Image.open("page2/image/unet++/Train_batch_cost.png"))
    with train_loss:
        st.image(Image.open("page2/image/unet++/Train_loss.png"))
    with reader_cost:
        st.image(Image.open("page2/image/unet++/Train_reader_cost.png"))
def AttentionUnet():
    st.header("3️⃣ AttentionUnet", help="No")
    mIoU, Acc, mRecall, batch_cost, train_loss, reader_cost = \
        st.tabs(["mIoU", "Acc", "mRecall",  "batch_cost", "train_loss","reader_cost"])
    with mIoU:
        st.image(Image.open("page2/image/attention/Evaluate_mIoU.png"))
    with Acc:
        st.image(Image.open("page2/image/attention/Evaluate_Acc.png"))
    with mRecall:
        st.image(Image.open("page2/image/attention/Evaluate_mRecall.png"))
    with batch_cost:
        st.image(Image.open("page2/image/attention/Train_batch_cost.png"))
    with train_loss:
        st.image(Image.open("page2/image/attention/Train_loss.png"))
    with reader_cost:
        st.image(Image.open("page2/image/attention/Train_reader_cost.png"))

def Fcn():
    st.header("4️⃣ Fcn", help="No")
    mIoU, Acc, mRecall, batch_cost, train_loss, reader_cost = \
        st.tabs(["mIoU", "Acc", "mRecall",  "batch_cost", "train_loss","reader_cost"])
    with mIoU:
        st.image(Image.open("page2/image/Fcn/Evaluate_mIoU.png"))
    with Acc:
        st.image(Image.open("page2/image/Fcn/Evaluate_Acc.png"))
    with mRecall:
        st.image(Image.open("page2/image/Fcn/Evaluate_mRecall.png"))
    with batch_cost:
        st.image(Image.open("page2/image/Fcn/Train_batch_cost.png"))
    with train_loss:
        st.image(Image.open("page2/image/Fcn/Train_loss.png"))
    with reader_cost:
        st.image(Image.open("page2/image/Fcn/Train_reader_cost.png"))

def PPUnet():
    st.header("🎖️ MTDUnet", help="No")
    mIoU, Acc, mRecall, batch_cost, train_loss, reader_cost = \
        st.tabs(["mIoU", "Acc", "mRecall",  "batch_cost", "train_loss","reader_cost"])
    with mIoU:
        st.image(Image.open("page2/image/PPUnet/Evaluate_mIoU.png"))
    with Acc:
        st.image(Image.open("page2/image/PPUnet/Evaluate_Acc.png"))
    with mRecall:
        st.image(Image.open("page2/image/PPUnet/Evaluate_mRecall.png"))
    with batch_cost:
        st.image(Image.open("page2/image/PPUnet/Train_batch_cost.png"))
    with train_loss:
        st.image(Image.open("page2/image/PPUnet/Train_loss.png"))
    with reader_cost:
        st.image(Image.open("page2/image/PPUnet/Train_reader_cost.png"))

left_column_defaults = [
    "1️⃣ Unet",
    "2️⃣ Unet++",
    "3️⃣ AttentionUnet",
    # "🗃 schema",
    # "📊 table",
    # "🔎 view",
    # "📸 materialized view",
    # "🔄 dynamic table",
    # "📋 task",
    # "🌊 stream",
    # "🚨 alert",
]

right_column_defaults = [
    "4️⃣ Fcn",
    "🎖️ MTDUnet"
    # "🚉 stage",
    # "🚚 data loading",
    # "🌀 data manipulation",
    # "🪄 function",
    # "🪜 procedure",
    # "🚰 pipe",
]
# left_col_container = st.empty()
# right_col_container = st.empty()
# all_segments = left_column_defaults + right_column_defaults
# side_left_col, side_right_col = st.columns(2)
# # left_col_segments = side_left_col.multiselect("Left Column",
# #                                                   options=all_segments,
# #                                                   default=left_column_defaults,
# #                                                   key="layout_left_column")
# left_col_segments = left_col_container.multiselect("Left Column",
#                                                   options=all_segments,
#                                                   default=left_column_defaults,
#                                                   key="layout_left_column")
# right_col_segments = right_col_container.multiselect("Right Column",
#                                                     options=all_segments,
#                                                     default=right_column_defaults,
#                                                     key="layout_right_column")

segment_dict = {
    "1️⃣ Unet": Unet,
    "2️⃣ Unet++": Unetpp,
    "3️⃣ AttentionUnet": AttentionUnet,
    "4️⃣ Fcn": Fcn,
    "🎖️ MTDUnet": PPUnet,
    # "📊 table": table_segment,
    # "🗃 schema": schema_segment,
    # "🔎 view": view_segment,
    # "📸 materialized view": materialized_view_segment,
    # "🚉 stage": stage_segment,
    # "🚰 pipe": pipe_segment,
    # "🚚 data loading": data_loading_segment,
    # "📋 task": task_segment,
    # "🌊 stream": stream_segment,
    # "🪄 function": function_segment,
    # "🪜 procedure": procedure_segment,
    # "🔄 dynamic table": dynamic_table_segment,
    # "🚨 alert": alert_segment,
    # "🌀 data manipulation": data_manipulation_segment,

}
with cols[0]:
    for seg in left_column_defaults:
        segment_dict[seg]()

with cols[1]:
    for seg in right_column_defaults:
        segment_dict[seg]()

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
