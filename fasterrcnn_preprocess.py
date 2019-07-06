import faster_rcnn.faster_rcnn as fr
import cv2
import numpy as np
import json

faster_model = fr.FasterRCNN()
# # output
# {
#     '000000035005.jpg': ## key: image path
#         [
#             (   
#                 'person', ## class
#                 [ 480.0917,  200.1376,  576.3452,  402.0709,    0.9994]
#                 ## list: [left_upper x, y, right lower x, y, confidence]
#             ),
#             (   
#                 'backpack', 
#                 [ 513.0871,  205.2412,  586.1628,  290.5187,    0.9915]
#             )
#         ],
#     'blablabla.jpg':
#         [
#         ]
# }
        

# = = = = = =
print('start processing testing data')
test = faster_model.process('data/hico_20150920/images/test2015')
with open(f'data/test_bbox.json', 'w') as f:
    json.dump(test, f)

print('start processing training data')
train = faster_model.process('data/hico_20150920/images/train2015')
with open(f'data/train_bbox.json', 'w') as f:
    json.dump(train, f)

# = = = = = =

# # read image using cv2
# data = cv2.imread('input_img/000000035005.jpg')
# bbox = result[0][1][0]
# t = tuple(int(np.round(x)) for x in bbox[1][:4])

# # draw bounding box
# cv2.rectangle(data, t[0:2], t[2:4], (0, 204, 0), 2)
# # crop object
# data = data[t[1]:t[3], t[0]:t[2]]
# # save image
# cv2.imwrite('test.png', data)

