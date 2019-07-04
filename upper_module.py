import json

def test(person, bbox): # not sure the format of bbox
    cnt = 0
    print(len(person['joints']))
    for i in range(17):
        x = person['joints'][i*3]
        y = person['joints'][i*3+1]
        print(x, y)

'''
        if point in bbox:
            cnt += 1
    if cnt >= 15
        return True

def cut(person)
    target = []
    res = []
    for i in range(17):
        x = person['keypoints'][j*3]
        y = person['keypoints'][j*3+1]
        if point in bbox:
'''
with open('results.json') as json_file:
    pics = json.load(json_file)
    pic = pics['000000035005.jpg']
    people = pic['bodies']
    for i, person in enumerate(people):
        print(len(person))
        test(person, [0, 0, 0, 0])
        # if test(person, bbox):
        #     return cut(person)

