from fpdf import FPDF

# 假设你已经有了图片和文本数据
# print(f"试图打开的文件路径: {image_path}")
# text_data = "这里是相关的文本数据"
# img = Image.open(image_path)
# 创建一个PDF对象
pdf = FPDF()
pdf.add_page()
pdf.add_font('HeiTi', '', 'style/Hei.ttf', uni=True)
pdf.add_font('SongTi', '', 'style/Song.ttf', uni=True)
# 添加图片到PDF
# pdf.image(image_path, x = 10, y = 10, w = 30,h=30)  # 调整x, y, w, h参数以适应你的需求
# 保存PDF到临时文件
# pdf_output = "../pdf/downloaded_file.pdf"
# pdf.output(pdf_output)
def converpdf(original,predict,level,area):

    pdf.set_font('HeiTi', size=30)
    pdf.cell(200, 10, txt="预估报告单", ln=True, align='C')
    cell_width = 50
    temp_width = 10
    # 显示 level 和 area 并排
    pdf.set_font('HeiTi', size=20)
    pdf.cell(cell_width, 10, txt="健康等级：", align='C')
    pdf.set_font('SongTi', size=15)
    pdf.cell(temp_width, 10, txt=level, ln=True,align='C')

    pdf.set_font('HeiTi', size=20)
    pdf.cell(cell_width, 10, txt="预估面积：", align='C')
    pdf.set_font('Arial', size=15)
    pdf.cell(temp_width, 10, txt=f"{str(round(area, 3))} cm^2", align='C')

    pdf.image(original, x=10, y=60, w=80, h=80)
    pdf.image(predict, x=120, y=60, w=80, h=80)

    pdf.set_y(267)  # 设置 y 坐标为页面底部
    pdf.set_font('SongTi', size=12)
    pdf.cell(0, 10, txt="2000300309陈乐设计", align='C')

    pdf_output = "pdf/file.pdf"
    pdf.output(pdf_output)
    return  pdf_output
