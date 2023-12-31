import cv2
from PIL import Image
from detecto.core import Model
from detecto import utils
import numpy as np
import streamlit as st
from utils import classify, save_file

model = 'vgg_16.h5'
class_names = ['Command', 'Crosswalk', 'No Entry', 'Speed Limit', 'Warning']

def main():
	new_title = '<p style = "text-align: center; font-weight: 1000; font-size: 29px;">ỨNG DỤNG NHẬN DẠNG</p>'
	st.markdown(new_title, unsafe_allow_html = True)
	new_title = '<p style = "text-align: center; font-weight: 1000; font-size: 29px;">5 ĐỐI TƯỢNG BIỂN BÁO GIAO THÔNG</p>'
	st.markdown(new_title, unsafe_allow_html = True)

	menu = ["Giới thiệu bài toán", "Phân loại hình ảnh", "Phát hiện vị trí", ]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Giới thiệu bài toán":
		new_title = '<p style = "text-align: center; font-weight: 1000; color: #FF847A; font-size: 27px;">MÔ TẢ BÀI TOÁN</p>'
		st.markdown(new_title, unsafe_allow_html = True)
		st.write("")
		st.write("Nhận dạng biển báo giao thông là vấn đề quan trọng vì nó hỗ trợ người tài xế ý thức và chủ động hơn trong")
		st.write("việc xử lý các tình huống nguy hiểm tiềm ẩn khi điều khiển phương tiện lưu thông.")
		st.write("Có hơn 1800 biển báo giao thông khác nhau, nhưng ở nghiên cứu này chỉ thực hiện nhận dạng 5 loại biển")
		st.write("báo giao thông khác nhau. Cụ thể là ở nghiên cứu này, tôi sẽ thực hiện nhận dạng 5 loại biển báo:")
		st.image('Untitled.png')

	elif choice == "Phân loại hình ảnh":
		new_title = '<p style = "text-align: center; font-weight: 1000; color: #FF847A; font-size: 27px;">PHÂN LOẠI 5 LOẠI BIỂN BÁO</p>'
		st.markdown(new_title, unsafe_allow_html = True)
		st.text("_______________Command - Crosswalk - No Entry - Speed Limit - Warning_______________")
		uploaded_file = st.file_uploader('', type = ['jpeg', 'jpg', 'png', 'gif'])

		if uploaded_file is not None:
			image = Image.open(uploaded_file)
			st.image(image, use_column_width = True)
			prediction, label = classify(image, model, class_names)
			st.write(f"<div style = 'text-align: center'><h1>{label}</h1></div>", unsafe_allow_html = True)

	else:
		new_title = '<p style = "text-align: center; font-weight: 1000; color: #FF847A; font-size: 27px;">XÁC ĐINH VỊ TRÍ 5 LOẠI BIỂN BÁO</p>'
		st.markdown(new_title, unsafe_allow_html = True)
		st.text("_______________Command - Crosswalk - No Entry - Speed Limit - Warning_______________")
		image_file = st.file_uploader('', type = ['jpeg', 'jpg', 'png', 'gif'])

		if image_file is not None:
			tmp_path = save_file(image_file)
			image = utils.read_image(tmp_path)
			model_dec = Model.load('detection.pth', ['command', 'crosswalk', 'no_entry', 'speed_limit', 'warning'])
			labels, boxes, scores = model_dec.predict(image)

			filter_all = np.where(scores > 0.9)
			filter_boxes = boxes[filter_all]
			print(scores)
			for i, box in enumerate(filter_boxes):
				x1, y1, x2, y2 = box

				cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

				cv2.putText(image, "", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

			st.image(image, use_column_width = True)

if __name__ == '__main__':
	main()