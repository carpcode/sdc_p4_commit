import cv2
import matplotlib.pyplot as plt

image = cv2.imread('tmp_data/data/IMG/center_2016_12_01_13_30_48_287.jpg')

f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(image)
ax2.imshow(cv2.flip(image,1))
f.savefig('resources/flip.jpeg')
plt.show()

'''
for i in range(3):
    print(i)
'''