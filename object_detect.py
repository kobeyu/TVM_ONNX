import os
import argparse

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

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

def LoadTFLiteInterpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def GetInputImage(path):
    if None == path:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imwrite('input.jpg', frame)
        return frame
    else:
        return  cv2.imread(path)

def PreProcess(interpreter, input_img):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
    img = img.astype(np.uint8)


    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], img)

def BBX2Str(bbx):
    _str = " ".join(str(bbx).split()) #remove multiple space
    return _str.replace('[','').replace(']','')

def SendStrToArduino(str):
    print(str)

def PostProcessByArduino(interpreter, input_image):
    BBX_NUM_IDX = 3
    CLS_IDX = 1
    SCR_IDX = 2
    BBX_IDX = 0
    output_details =interpreter.get_output_details()

    bbx_num = int(interpreter.get_tensor(output_details[BBX_NUM_IDX]['index'])[0])
    _class = interpreter.get_tensor(output_details[CLS_IDX]['index'])
    _score = interpreter.get_tensor(output_details[SCR_IDX]['index'])
    _bbx = interpreter.get_tensor(output_details[BBX_IDX]['index'])

    all_bbx = np.concatenate((_class.reshape((bbx_num, 1)), _score.reshape((bbx_num, 1)), _bbx.reshape((bbx_num, 4))), axis=1)

    bbx_str =''
    for bbx in all_bbx:
        bbx_str += BBX2Str(bbx)

    SendStrToArduino(bbx_str)

def PostProcess(interpreter, input_image):
    output_details =interpreter.get_output_details()
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

    scores = interpreter.get_tensor(output_details[2]['index'])
    print("score: ")
    print(scores)
    print(scores.shape)
    print(type(scores))


    num = interpreter.get_tensor(output_details[3]['index'])
    print("num:")
    print(num)

    fontPath = "./arial.ttf"
    font = ImageFont.truetype(fontPath, 192)

    for i in range(boxes.shape[1]):
        if scores[0, i] > 0.55:
            box = boxes[0, i, :]
            x0 = int(box[1] * input_image.shape[1])
            y0 = int(box[0] * input_image.shape[0])
            x1 = int(box[3] * input_image.shape[1])
            y1 = int(box[2] * input_image.shape[0])
            box = box.astype(np.int)
            cv2.rectangle(input_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(input_image, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)

            print("tflite " + label2string[int(labels[0, i])])

            cv2.putText(input_image,
                label2string[int(labels[0, i])],
                (x0, y0),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                2)
		# if tvm_output_score[0, i] > 0.7:
		# 	print("tvm " + label2string[int(tvm_output_label[0, i])])

    cv2.imwrite('output.jpg', input_image)
	# cv2.imshow('image', input_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



def Inference(interpreter):
    interpreter.invoke()


def InitArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="File path of tflite model", default="detect.tflite")

    parser.add_argument("--input", "-i", help="Input file jpg or video, default input data from camera")
    parser.add_argument("--target", "-t", help="Hardware target for post-process, 0:x86(default), 1:arduino", type=int, choices=[0,1], default=0)
    return parser.parse_args()


def main(args):
    interpreter = LoadTFLiteInterpreter(args.model)

    input_img = GetInputImage(args.input);
    PreProcess(interpreter, input_img)
    Inference(interpreter)

    if 0 == args.target:
        PostProcess(interpreter, input_img)
    else:
        PostProcessByArduino(interpreter, input_img)

if __name__ == "__main__":
    args = InitArgParser()

    main(args)

