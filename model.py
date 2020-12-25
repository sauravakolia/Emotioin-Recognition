import  cv2
import numpy as np 

# import requests

from fastai.vision.all import *
# from fastai.metrics import error_rate, accuracy

from emotion import EmotionDataset2,EmotionImage

from pathlib import Path

# from pathlib import Path
import pickle

import pathlib
# import torch
# torch.cuda.device("cpu")



# from emotion.ipynb import *
pathlib.PosixPath = pathlib.WindowsPath


# class EmotionImage(fastuple):
#     def show(self, ctx=None, **kwargs):
#         img, emotion = self
#         if not isinstance(img, Tensor):
#           image=tensor(img).float()/255.0
#           image=image.unsqueeze(0)
#           image=image.expand(3,48,48)
          
#           img_tensor = image
#         else:
#           img_tensor = img
#         return show_image(img_tensor,title=emotion, ctx=ctx, **kwargs)


objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

learn=load_learner('export.pkl',cpu=True)
# learn=learn.model.cpu()
# learn=learn.dls.cpu()
# learn.model.to('cpu')


class face_detector(object):

	def __init__(self,image_path=None):

		if image_path == None :
			self.video=cv2.VideoCapture(0)

		else:
			self.image=cv2.imread(image_path)

		self.face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

	# def __del__(self):
	# 	self.video.release()



	def webcam(self):
		

		while True:
			_,image=self.video.read()
			# image=cv2.flip(image,180)

			faces=self.face_cascade.detectMultiScale(image,1.3,5)

			for (x,y,w,h) in faces:
				cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

				roi=image[y:y+h,x:x+w]

				text=self.prediction(roi)

				cv2.putText(image,text , (x+5,y+5) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),1)

			cv2.imshow("img",image)
			# cv2.waitKey(0)

			if cv2.waitKey(1) & 0xFF == ord('q'): break

		self.video.release()
		cv2.destroyAllWindows()




	def picture(self):
		img=self.image
		faces=self.face_cascade.detectMultiScale(img,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

			roi=img[y:y+h,x:x+w]

			text=self.prediction(roi)


			cv2.putText(img, text , (x+5,y+5) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),1)

		cv2.imshow("img",img)
		cv2.waitKey(0)

		
	def prediction(self,roi):
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
		gray=cv2.resize(gray,(48,48), interpolation =cv2.INTER_AREA)
		im_pil = Image.fromarray(gray)
		img=PILImage(im_pil)
		# img=img.resize((28,28))
		k=EmotionImage(img)
		pred=learn.predict(k)

		p=torch.argmax(pred[0])

		text=objects[p.item()]
		print(text)

		

		return text


		



f=face_detector("download (1).jpg")

f.picture()

# f.webcam()
	# key = cv2.waitKey(1) & 0xFF

	# if key == ord("q"):
	# 	video.release()
	# 	break

	