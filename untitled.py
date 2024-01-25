# -*- coding: utf-8 -*-


Original file is located at
    https://colab.research.google.com/drive/1Ze4LGiyGCvomlUIOiOTC3C1AgDuaXl05
"""

pip install deepface

!git clone https://github.com/Abhiparouha/Emotion.git

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1=cv2.imread('/content/Emotion/road line detection/OIP.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()

result=DeepFace.analyze(img1, actions=['emotion'])

print(result)
