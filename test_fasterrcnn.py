import faster_rcnn.faster_rcnn as fr
import cv2
import numpy as np

faster_model = fr.FasterRCNN()
result = faster_model.process('input_img') #input: input images folder
# # output
# [   
#     (   
#         'input_img/000000035005.jpg', ## image path
#         [
#             (   
#                 'person', ## class
#                 tensor([ 480.0917,  200.1376,  576.3452,  402.0709,    0.9994])
#                 ## [left_upper x, lu y, right lower x, rl y, confidence]
#             ),
#             (   
#                 'backpack', 
#                 tensor([ 513.0871,  205.2412,  586.1628,  290.5187,    0.9915])
#             )
#         ]
#     ),
#     (
#     )
# ]

# = = = = = =

# print bounding box information
# for i in result:
# 	for j in i:
#         print(j)

# = = = = = =

# read image using cv2
data = cv2.imread('input_img/000000035005.jpg')
bbox = result[0][1][0]
t = tuple(int(np.round(x)) for x in bbox[1][:4])

# draw bounding box
cv2.rectangle(data, t[0:2], t[2:4], (0, 204, 0), 2)
# crop object
data = data[t[1]:t[3], t[0]:t[2]]
# save image
cv2.imwrite('test.png', data)

