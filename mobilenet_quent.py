# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tflite

# https://www.tensorflow.org/lite/guide/hosted_models
# http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

label2string = \
{
	0:   "person",
	1:   "bicycle",
	2:   "car",
	3:   "motorcycle",
	4:   "airplane",
	5:   "bus",
	6:   "train",
	7:   "truck",
	8:   "boat",
	9:   "traffic light",
	10:  "fire hydrant",
	12:  "stop sign",
	13:  "parking meter",
	14:  "bench",
	15:  "bird",
	16:  "cat",
	17:  "dog",
	18:  "horse",
	19:  "sheep",
	20:  "cow",
	21:  "elephant",
	22:  "bear",
	23:  "zebra",
	24:  "giraffe",
	26:  "backpack",
	27:  "umbrella",
	30:  "handbag",
	31:  "tie",
	32:  "suitcase",
	33:  "frisbee",
	34:  "skis",
	35:  "snowboard",
	36:  "sports ball",
	37:  "kite",
	38:  "baseball bat",
	39:  "baseball glove",
	40:  "skateboard",
	41:  "surfboard",
	42:  "tennis racket",
	43:  "bottle",
	45:  "wine glass",
	46:  "cup",
	47:  "fork",
	48:  "knife",
	49:  "spoon",
	50:  "bowl",
	51:  "banana",
	52:  "apple",
	53:  "sandwich",
	54:  "orange",
	55:  "broccoli",
	56:  "carrot",
	57:  "hot dog",
	58:  "pizza",
	59:  "donut",
	60:  "cake",
	61:  "chair",
	62:  "couch",
	63:  "potted plant",
	64:  "bed",
	66:  "dining table",
	69:  "toilet",
	71:  "tv",
	72:  "laptop",
	73:  "mouse",
	74:  "remote",
	75:  "keyboard",
	76:  "cell phone",
	77:  "microwave",
	78:  "oven",
	79:  "toaster",
	80:  "sink",
	81:  "refrigerator",
	83:  "book",
	84:  "clock",
	85:  "vase",
	86:  "scissors",
	87:  "teddy bear",
	88:  "hair drier",
	89:  "toothbrush",
}




def detect_from_image():
	# prepara input image
	# img_org = cv2.imread('person.png')
	# img_org = cv2.imread('person.png')
	img_org = cv2.imread('hqdefault.jpg')

	print(img_org.shape)

	#cv2.imshow('image', img)
	img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (300, 300))
	img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
	img = img.astype(np.uint8)

	input_tensor = "normalized_input_image_tensor"
	input_shape = (1, 300, 300, 3)
	input_dtype = "uint8"

	# TVM load
	tflite_model_buf = open("detect.tflite", "rb").read()
	tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

	# Parse TFLite model and convert it to a Relay module
	from tvm import relay, transform

	mod, params = relay.frontend.from_tflite(
		tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
	)
	target = "llvm"
	with transform.PassContext(opt_level=3):
		lib = relay.build(mod, target, params=params)


	# TVM Run
	import tvm
	from tvm import te
	from tvm.contrib import graph_executor as runtime

	# Create a runtime executor module
	module = runtime.GraphModule(lib["default"](tvm.cpu()))

	# Feed input data
	module.set_input(input_tensor, tvm.nd.array(img))

	# Run
	module.run()

	# Get output
	tvm_output_box = module.get_output(0).numpy()

	# print(tvm_output_box)

	tvm_output_label = module.get_output(1).numpy()
	# print(tvm_output_label)

	tvm_output_score = module.get_output(2).numpy()
	# print(tvm_output_score)




	# TFLITE
	# load model
	interpreter = tf.lite.Interpreter(model_path="detect.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	# set input tensor
	interpreter.set_tensor(input_details[0]['index'], img)

	# run
	interpreter.invoke()

	# get outpu tensor
	boxes = interpreter.get_tensor(output_details[0]['index'])
	print("Box:")
	print(boxes)
	print("box shape")
	print(boxes.shape)
	print(type(boxes))
	labels = interpreter.get_tensor(output_details[1]['index'])
	print("Label:")
	print(labels)
	print(type(labels))
	print(labels.shape)
	# print(output_details[1])
	# print(output_details)
	# print(boxes.shape[1])
	
	scores = interpreter.get_tensor(output_details[2]['index'])
	print("score: ")
	print(scores)
	print(scores.shape)
	print(type(scores))


	num = interpreter.get_tensor(output_details[3]['index'])
	# print(num)

	fontPath = "./arial.ttf"
	font = ImageFont.truetype(fontPath, 192)

	for i in range(boxes.shape[1]):
		# print(scores[0, i])
		if scores[0, i] > 0.7:
			box = boxes[0, i, :]
			x0 = int(box[1] * img_org.shape[1])
			y0 = int(box[0] * img_org.shape[0])
			x1 = int(box[3] * img_org.shape[1])
			y1 = int(box[2] * img_org.shape[0])
			box = box.astype(np.int)
			cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
			cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)

			print("tflite " + label2string[int(labels[0, i])])

			cv2.putText(img_org,
				   label2string[int(labels[0, i])],
				   (x0, y0),
				   cv2.FONT_HERSHEY_COMPLEX_SMALL,
				   1,
				   (255, 255, 255),
				   2)
		
		# if tvm_output_score[0, i] > 0.7:
		# 	print("tvm " + label2string[int(tvm_output_label[0, i])])

	cv2.imwrite('output.jpg', img_org)
	# cv2.imshow('image', img_org)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()




def detect_from_video():
	# prepara input image
	# img_org = cv2.imread('person.png')
	# img_org = cv2.imread('person.png')
	# img_org = cv2.imread('hqdefault.jpg')

	#cv2.imshow('image', img)
	# img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	# img = cv2.resize(img, (300, 300))
	# img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
	# img = img.astype(np.uint8)

	cap = cv2.VideoCapture('test_video_Trim.mp4')
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')

	ret, frame = cap.read()



	out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame.shape[1], frame.shape[0]))

	input_tensor = "normalized_input_image_tensor"
	input_shape = (1, 300, 300, 3)
	input_dtype = "uint8"

	# TVM load
	tflite_model_buf = open("detect.tflite", "rb").read()
	tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

	# Parse TFLite model and convert it to a Relay module
	from tvm import relay, transform

	mod, params = relay.frontend.from_tflite(
		tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
	)
	target = "llvm"
	with transform.PassContext(opt_level=3):
		lib = relay.build(mod, target, params=params)


	# TVM Run
	import tvm
	from tvm import te
	from tvm.contrib import graph_executor as runtime

	# Create a runtime executor module
	module = runtime.GraphModule(lib["default"](tvm.cpu()))

	frame_num = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		
		print(frame.shape)

		img_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img_tmp = cv2.resize(img_tmp, (300, 300))

		# cv2.imwrite("output/out" + str(frame_num) + ".jpg", img_tmp)
		# frame_num += 1
		# continue
		img = img_tmp.reshape(1, img_tmp.shape[0], img_tmp.shape[1], img_tmp.shape[2]) # (1, 300, 300, 3)
		img = img.astype(np.uint8)

		# cv2.imwrite("output/out" + str(frame_num) + ".jpg", img_tmp)
		# frame_num += 1
		# continue

		# Feed input data
		module.set_input(input_tensor, tvm.nd.array(img))
		
		print(img)
		print(img.shape)
		

		# Run
		module.run()

		# Get output
		tvm_output_box = module.get_output(0).numpy()

		# print(tvm_output_box)

		tvm_output_label = module.get_output(1).numpy()
		# print(tvm_output_label)

		tvm_output_score = module.get_output(2).numpy()
		# print(tvm_output_score)




		# TFLITE
		# load model
		interpreter = tf.lite.Interpreter(model_path="detect.tflite")
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		# set input tensor
		interpreter.set_tensor(input_details[0]['index'], img)

		# run
		interpreter.invoke()

		# get outpu tensor
		boxes = interpreter.get_tensor(output_details[0]['index'])
		print("Box:")
		print(boxes)
		print("box shape")
		print(boxes.shape)
		labels = interpreter.get_tensor(output_details[1]['index'])
		# print("Label:")
		# print(labels)
		# print(output_details[1])
		# print(output_details)
		# print(boxes.shape[1])
		
		scores = interpreter.get_tensor(output_details[2]['index'])
		# print(scores)
		num = interpreter.get_tensor(output_details[3]['index'])
		# print(num)

		fontPath = "./arial.ttf"
		font = ImageFont.truetype(fontPath, 192)

		img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)

		for i in range(boxes.shape[1]):
			# print(scores[0, i])
			if scores[0, i] > 0.7:
				box = boxes[0, i, :]
				x0 = int(box[1] * frame.shape[1])
				y0 = int(box[0] * frame.shape[0])
				x1 = int(box[3] * frame.shape[1])
				y1 = int(box[2] * frame.shape[0])
				box = box.astype(np.int)
				cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
				cv2.rectangle(frame, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)

				print("tflite " + label2string[int(labels[0, i])])

				cv2.putText(frame,
					label2string[int(labels[0, i])],
					(x0, y0),
					cv2.FONT_HERSHEY_COMPLEX_SMALL,
					1,
					(255, 255, 255),
					2)
			
			# if tvm_output_score[0, i] > 0.7:
			# 	print("tvm " + label2string[int(tvm_output_label[0, i])])

		print("frame num", frame_num)

		cv2.imwrite("output/out" + str(frame_num) + ".jpg", frame)
		# cv2.imshow('image', img_org)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		out.write(frame)

		frame_num += 1
		
	
	out.release()
	cap.release()
	# cv2.destroyAllWindows()


def test_video():
	cap = cv2.VideoCapture(-1)
	# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	
	ret, frame = cap.read()

	cv2.imwrite("test.jpg", frame)

	# out = cv2.VideoWriter('output.mp4', fourcc, 20, (300,300))

	# frame_num = 0

	# while True:
	# 	ret, frame = cap.read()

	# 	# cv2.imwrite("1.jpg", frame)
	# 	# break
	# 	if not ret:
	# 		break
		
	# 	print(frame.shape)
	# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# 	frame = cv2.resize(frame, (300, 300))
	# 	img = frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2]) # (1, 300, 300, 3)
	# 	img = img.astype(np.uint8)
		
	# 	print(img)
	# 	print(img.shape)
	# 	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	# 	cv2.imwrite("2.jpg", frame)

	# 	frame_num += 1
	# 	break


if __name__ == '__main__':
	# test_video()
	detect_from_image()
	detect_from_video()