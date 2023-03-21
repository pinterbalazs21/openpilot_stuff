import cv2
import json
import numpy as np
import onnxruntime
import pandas as pd
from matplotlib import pyplot as plt
from inputWrapper import InputWrapper
from outputWrapper import OutputWrapper, ModelOutputVisualizer

X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

def parse_image(frame):
	print(type(frame))
	H = (frame.shape[0]*2)//3
	W = frame.shape[1]
	print("H, W:" + str(H) + "; " +  str(W))
	parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)

	parsed[0] = frame[0:H:2, 0::2]
	parsed[1] = frame[1:H:2, 0::2]
	parsed[2] = frame[0:H:2, 1::2]
	parsed[3] = frame[1:H:2, 1::2]
	parsed[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
	parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))

	return parsed

def seperate_points_and_std_values(df):
	points = df.iloc[lambda x: x.index % 2 == 0]
	std = df.iloc[lambda x: x.index % 2 != 0]
	points = pd.concat([points], ignore_index = True)
	std = pd.concat([std], ignore_index = True)

	return points, std

def apply_perspective_transform(points, src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_points = cv2.perspectiveTransform(points, M)
    return transformed_points

# Camera parameters
camera_height = 1.5
camera_fov = 105

def main():
	model = "supercombo.onnx"
	
	cap = cv2.VideoCapture('dashcam.mp4')
	parsed_images = []

	width = 512
	height = 256
	dim = (width, height)	
	
	session = onnxruntime.InferenceSession(model, None)
	idx = 0
	visualizer = ModelOutputVisualizer()
	while(cap.isOpened()):

		ret, frame = cap.read()
		if (ret == False):
			break

		if frame is not None:
			img = cv2.resize(frame, dim)
			img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
			parsed = parse_image(img_yuv)
	
		if (len(parsed_images) >= 2):
			del parsed_images[0]
	
		parsed_images.append(parsed)

		if (len(parsed_images) >= 2):		
			parsed_arr = np.array(parsed_images)
			parsed_arr.resize((1,12,128,256))
			data = json.dumps({'data': parsed_arr.tolist()})
			data = np.array(json.loads(data)['data']).astype('float32')
			output_name = session.get_outputs()[0].name

			print("--------------------")
			input_wrapper = InputWrapper(session, data, data)
			result = session.run([output_name], input_wrapper.get_model_input())
			model_output = OutputWrapper(result)
			visualizer.visualize(frame, model_output)
		

		frame = cv2.resize(frame, (900, 500))
		#cv2.imshow('frame', frame)
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#	break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
