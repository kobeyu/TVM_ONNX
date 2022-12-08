import os
import argparse
import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import serial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

MODEL_INPUT_W = 300
MODEL_INPUT_H = 300

port = "/dev/ttyUSB1"

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

def GetImageFromCamera(cap):
        ret, frame = cap.read()
        cv2.imwrite('camera.jpg', frame)
        return  frame

def PreProcess(interpreter, input_img):
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (MODEL_INPUT_W, MODEL_INPUT_H))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
    img = img.astype(np.uint8)

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], img)

def BBX2Str(bbx):
    _str = " ".join(str(bbx).split()) #remove multiple space
    return _str.replace('[','').replace(']','')


def SendStrToArduino(data):
    ## TODO modify port

    dtype = data.dtype
    shape = data.shape
    # serialize bbx data and send to Andes board
    byte_data = data.tobytes()
    send_size = ser.write(byte_data)

    # Receive data from Andes board
    redeive_data = ser.read(send_size)
    assert len(redeive_data) == send_size , "Receive_data size should equal to send_data size"

    out_put = np.frombuffer(redeive_data, dtype=dtype).reshape(shape)
    print("Receive data", out_put.shape)
    return out_put


def PostProcess(interpreter, input_image, quan=True):
    BBX_NUM_IDX = 3
    CLS_IDX = 1
    SCR_IDX = 2
    BBX_IDX = 0
    output_details =interpreter.get_output_details()

    bbx_num = int(interpreter.get_tensor(output_details[BBX_NUM_IDX]['index'])[0])

    _class = interpreter.get_tensor(output_details[CLS_IDX]['index'])
    _score = interpreter.get_tensor(output_details[SCR_IDX]['index'])
    _bbx = interpreter.get_tensor(output_details[BBX_IDX]['index'])

    if quan:
        _class = _class.astype(np.int32)
        _score = (_score * 255).astype(np.int32)
        image_xy = [ input_image.shape[0], input_image.shape[1], input_image.shape[0], input_image.shape[1]]
        _bbx = (_bbx[::4] * image_xy).astype(np.int32)

    ori_all_bbx = np.concatenate((_class.reshape((bbx_num, 1)), _score.reshape((bbx_num, 1)), _bbx.reshape((bbx_num, 4))), axis=1)

    #apply relu operation, reduce to 8 bbx and transpose bbx data.
    all_bbx = np.maximum(0, np.delete(ori_all_bbx, [8,9], 0)).transpose()

    if 1 == args.target:
        all_bbx_back = SendStrToArduino(all_bbx)
    else:
        all_bbx_back = all_bbx.transpose()

    all_bbx_back.transpose()
    score_th = 125 if quan else 0.5

    for bbx in all_bbx_back:
        label = bbx[0]
        box = bbx[2:]
        score = bbx[1]
        if(score == -1 or score < score_th):
            continue
        if quan :
            x0 = box[1]
            y0 = box[0]
            x1 = box[3]
            y1 = box[2]
        else:
            x0 = int(box[1] * input_image.shape[1])
            y0 = int(box[0] * input_image.shape[0])
            x1 = int(box[3] * input_image.shape[1])
            y1 = int(box[2] * input_image.shape[0])

        cv2.rectangle(input_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.rectangle(input_image, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)

        #print("tflite " + label2string[int(label)])

        cv2.putText(input_image,
                label2string[int(label)],
                (x0, y0),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                2)
    return input_image


def e2e(interpreter, in_img, start_time, quan, wait_time = 0, video_wtr=None):
    PreProcess(interpreter, in_img)
    interpreter.invoke()
    out_img = PostProcess(interpreter, in_img, quan)
    ShowAndSaveResult(out_img, video_wtr)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        return 0
    return time.time() - start_time

def InitArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="File path of tflite model", default="model/detect.tflite")
    parser.add_argument("--input", "-i", help="path to jpg file,default input data from camera")
    parser.add_argument("--quan", "-q", action="store_true", help="Quantize model output")
    parser.add_argument("--target", "-t", help="Hardware target for post-process, 0:x86(default), 1:arduino", type=int, choices=[0,1], default=0)
    return parser.parse_args()


def GetCameraSource(ofile="output.mp4", width=640, heigh=480):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_wtr = cv2.VideoWriter(ofile, fourcc, 5, (width, heigh))
    return cap, video_wtr

def ShowAndSaveResult(output_image, video_wtr=None):
    cv2.imwrite('output.jpg', output_image)
    cv2.imshow('frame', output_image)
    if video_wtr:
        video_wtr.write(output_image)


def main(args):
    global ser
    interpreter = LoadTFLiteInterpreter(args.model)

    if 1 == args.target:
        ser = serial.Serial(port, 38400, timeout=20)

    if args.input:
        print("source data from jpg file")
        in_img = cv2.imread(args.input)
        e2e(interpreter, in_img, time.time(), args.quan)

    else:
        print("source data from camera")
        cap, video_wtr = GetCameraSource()
        last_time = time.time()
        while cap.isOpened():
            in_img = GetImageFromCamera(cap);
            diff = e2e(interpreter, in_img, time.time(), args.quan, 1, video_wtr)
            if not diff:
                break;

            print("fps:", 1.0/ diff)

    if not args.input:
        cap.release()
        video_wtr.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = InitArgParser()

    main(args)

