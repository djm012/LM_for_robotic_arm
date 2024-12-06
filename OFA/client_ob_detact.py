import cv2
import requests
import numpy as np
import time
import json
import os
from datetime import datetime
import urllib.request
import base64
import json_numpy
json_numpy.patch()

def show_welcome():
	print('='*50)
	print('Welcome to the xxx system, press enter to start!')
	input()

def get_command():
	print("\nPlease input the instruct(input 'quit' to exit):")
	return input().strip()

def save_data(frame, command,save_path='capture_data'):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

	img_path = f"{save_path}/image_{timestamp}.jpg"
	cv2.imwrite(img_path,frame)

	data = {
		"timestamp":timestamp,
		"command":command,
		"img_path":img_path
	}

	json_path = f"{save_path}/data_{timestamp}.json"
	with open(json_path,'w',encoding='utf-8') as f:
		json.dump(data,f,ensure_ascii=False,indent=4)


	print(f'data saved in {json_path}')


# vg: vision localtion model. large-general scene-english
def send2vg(image,command):
	para_data = {}
	para_data['text'] = command
	img = image
	file = {'file':("file_name.jpg",cv2.imencode(".jpg",img)[1].tobytes(),'image/jpg')}

	action = requests.post(
		"http://172.10.60.145:8000/vg",
		files=file,
		data=para_data
	)
	res = action.json()
	if res:
		print(res)
	else:
		print('dont find target location!')

# garbage_cls: the function of garbage classification
def send2garbage_cls(frame):
	url = "http://172.10.60.145:8000/garbage_cls"
    # url = "http://172.18.32.132:5001/garbage_cls"

	file = {"file": ("file_name.jpg", cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpg")}
	res = requests.post(url=url, files=file)
	res_data = res.json()
	print(res_data) 


def img2np(frame):
	frame_resized = cv2.resize(frame,(256,256))
	frame_rgb = cv2.cvtColor(frame_resized,cv2.COLOR_BGR2RGB)
	image_array = np.array(frame_rgb,dtype=np.uint8)
	# print(f'numpy shape is {image_array.shape}')
	return image_array



def showvideo():
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()
		cv2.imshow('show:',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()



def main():

	show_welcome()

	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print('can not open camera!')
		return

	# server_url = "http://172.17.0.1:5000/upload"
	command = get_command()

	while True:
		
		ret,frame = cap.read()
		# if command.lower() == 'quit':
		# 	print('exist the process....')
		# 	print('='*50)
		# 	break
		
		if not ret:
			print('can not get the picture')
			continue

		cv2.imshow('Camera',frame)
		cv2.waitKey(1)

		# send the img and command to api using urllib
		# action = send2api(img2np(frame),command)

		# # print('action:',action)
		# print(f"arm tail excecutor location:{'%.4f'%action[0],'%.4f'%action[1],'%.4f'%action[2]}")
		# print(f"arm tail excecutor current angle:{'%.4f'%action[3],'%.4f'%action[4],'%.4f'%action[5]}")
		# print(f"arm status:{'%.2f'%action[6]}")
		send2vg(frame,command)
		# send2garbage_cls(frame)
		print('='*50)


		# save_data(frame,command)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	# cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
