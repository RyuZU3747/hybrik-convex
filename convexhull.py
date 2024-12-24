import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import csv
import math
import copy
from math import atan2
from functools import cmp_to_key

def ccw(a, b, c):
   return a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - a[1]*b[0] - b[1]*c[0] - c[1]*a[0]

fp = [0,0]

def square_distance(point1, point2):
    return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2

def ccwsort(a, b):
    if ccw(fp, a, b):
        return 1
    return -1 

def convex_hull(points):
   points.sort()
   lower = []
   for p in points:
       while len(lower) >= 2 and ccw(lower[-2], lower[-1], p) < 0:
           lower.pop()
       lower.append(p)
  
   upper = []
   for p in reversed(points):
       while len(upper) >= 2 and ccw(upper[-2], upper[-1], p) < 0:
           upper.pop()
       upper.append(p)
      
   return lower[:-1] + upper[:-1]
  
def get_volume(points):
   ret = 0
   for i in range(1, len(points)-1):
       ret += abs((points[i][0] - points[0][0])*(points[i+1][1]-points[0][1]) - (points[i+1][0] - points[0][0])*(points[i][1]-points[0][1]))
   return ret/2

def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    def inside(p, cp1, cp2):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def compute_intersection(s, e, cp1, cp2):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for cp2 in clip_polygon:
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output_list.append(compute_intersection(s, e, cp1, cp2))
                output_list.append(e)
            elif inside(s, cp1, cp2):
                output_list.append(compute_intersection(s, e, cp1, cp2))
            s = e
        
        if len(output_list) == 0:
            return input_list
        # print(output_list)
            
        cp1 = cp2
        

    return output_list

def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def intersection_area(polygon1, polygon2):
    intersection_polygon = sutherland_hodgman_clip(polygon1, polygon2)
    if len(intersection_polygon) == 0:
        return 0.0
    return polygon_area(intersection_polygon)

def load_pickle_file(file_path):
   try:
       with open(file_path, 'rb') as file:
           obj = pickle.load(file)
       return obj
   except FileNotFoundError:
       print(f"파일 '{file_path}'를 찾을 수 없습니다.")
       return None
   except pickle.UnpicklingError:
       print(f"파일 '{file_path}'의 내용을 읽을 수 없습니다.")
       return None




file = "D:/research/HybrIK/seo.pk"


pk = load_pickle_file(file)


fig = plt.figure(1)
fig.canvas.draw()
ax = fig.add_subplot(111, projection="3d")
plt.xlim([-1, 1])
plt.ylim([-1, 1])

trlist = []
salist = []
colist = []
intlist = []
head = pk['pred_xyz_29'][0][12]
gnd = copy.deepcopy(head)
head[0] = 0
gnd[0] = 0
gnd[1] = 0
l = math.sqrt(head[1]**2 + head[2]**2) * gnd[2]
inn = head[1]*gnd[1] + head[2]*gnd[2]
cs = inn / l
pi = math.pi / 2
theta = math.acos(cs)
theta = pi - theta 
# theta = -theta
rotx = [[1.0, 0.0, 0.0],
      [0.0, math.cos(theta), -math.sin(theta)],
      [0.0, math.sin(theta), math.cos(theta)]]



for frame in range(0, len(pk['pred_xyz_29'])):
    transverse = []
    sagittal = []
    coronal = []
    legs = []
    ups = []
    leftleg = [[],[],[]]
    leftarm = [[],[],[]]
    rightleg = [[],[],[]]
    rightarm = [[],[],[]]
    spine = [[],[],[]]
    for i, points in enumerate(pk['pred_xyz_29'][frame]):
        points = rotx @ points
        if i in [0,1,4,7,10,27]:
            leftleg[0].append(points[0])
            leftleg[1].append(points[2])
            leftleg[2].append(-points[1])
            ax.scatter(points[0], points[2], -points[1])
        if i in [0,2,5,8,11,28]:
            rightleg[0].append(points[0])
            rightleg[1].append(points[2])
            rightleg[2].append(-points[1])
            ax.scatter(points[0], points[2], -points[1])
        if i in [0,3,6,9,12,24]:
            spine[0].append(points[0])
            spine[1].append(points[2])
            spine[2].append(-points[1])
            ax.scatter(points[0], points[2], -points[1])
        if i in [9,13,16,18,20,22,25]:
            leftarm[0].append(points[0])
            leftarm[1].append(points[2])
            leftarm[2].append(-points[1])
            ax.scatter(points[0], points[2], -points[1])
        if i in [9,14,17,19,21,23,26]:
            rightarm[0].append(points[0])
            rightarm[1].append(points[2])
            rightarm[2].append(-points[1])
            ax.scatter(points[0], points[2], -points[1])
        # if i in [0,3,6,9]:
        if i in [0,1,2,4,5,7,8,10,11,27,28]:
            legs.append([points[0], points[2]])
        # if i in [0,1,2,3,6,9,13,14]:
        if i not in [0,1,2,4,5,7,8,10,11,27,28,24]:
            ups.append([points[0], points[2]])
            # transverse.append([points[0], points[2]])
            # sagittal.append([points[1], points[2]])
            # coronal.append([points[0], points[1]])
        # if i in [0,1,2,3]:
        #     leftleg[0].append(points[0])
        #     leftleg[1].append(points[2])
        #     leftleg[2].append(-points[1])
        #     ax.scatter(points[0], points[2], -points[1])
        # if i in [0,4,5,6]:
        #     rightleg[0].append(points[0])
        #     rightleg[1].append(points[2])
        #     rightleg[2].append(-points[1])
        #     ax.scatter(points[0], points[2], -points[1])
        # if i in [0,7,8,9,10]:
        #     spine[0].append(points[0])
        #     spine[1].append(points[2])
        #     spine[2].append(-points[1])
        #     ax.scatter(points[0], points[2], -points[1])
        # if i in [8,11,12,13]:
        #     leftarm[0].append(points[0])
        #     leftarm[1].append(points[2])
        #     leftarm[2].append(-points[1])
        #     ax.scatter(points[0], points[2], -points[1])
        # if i in [8,14,15,16]:
        #     rightarm[0].append(points[0])
        #     rightarm[1].append(points[2])
        #     rightarm[2].append(-points[1])
        #     ax.scatter(points[0], points[2], -points[1])
        # if i in [2,3,5,6,9,10,12,13,15,16]:
        #     continue
        # p.append([points[0], points[2]])

    legs_ch = convex_hull(legs)
    legs_x = []
    legs_y = []
    legs_z = []
    for c in legs_ch:
        legs_x.append(c[0])
        legs_y.append(0)
        legs_z.append(c[1])
    legs_x.append(legs_ch[0][0])
    legs_y.append(0)
    legs_z.append(legs_ch[0][1])
    # print(frame, get_volume(legs_ch))
    trlist.append(get_volume(legs_ch))
    ax.plot(legs_x,legs_z,legs_y)
    
    ups_ch = convex_hull(ups)
    ups_x = []
    ups_y = []
    ups_z = []
    for c in ups_ch:
        ups_x.append(c[0])
        ups_y.append(0)
        ups_z.append(c[1])
    ups_x.append(ups_ch[0][0])
    ups_y.append(0)
    ups_z.append(ups_ch[0][1])
    # print(frame, get_volume(ups_ch))
    trlist.append(get_volume(ups_ch))
    ax.plot(ups_x,ups_z,ups_y)
    
    ups_ch.sort(key=lambda x: (x[1],x[0]))
    fp = ups_ch[0]
    ups_ch.remove(fp)
    ups_ch.sort(key=lambda x: (atan2(x[1] - fp[1], x[0] - fp[0]), square_distance(fp, x)))
    ccwups = [fp]
    for e in ups_ch:
        ccwups.append(e)

    legs_ch.sort(key=lambda x: (x[1],x[0]))
    fp = legs_ch[0]
    legs_ch.remove(fp)
    legs_ch.sort(key=lambda x: (atan2(x[1] - fp[1], x[0] - fp[0]), square_distance(fp, x)))
    ccwlegs = [fp]
    for e in legs_ch:
        ccwlegs.append(e)    
    
    intersection = sutherland_hodgman_clip(ccwups, ccwlegs)
    intersection_x = []
    intersection_y = []
    intersection_z = []
    for c in intersection:
        intersection_x.append(c[0])
        intersection_y.append(0)
        intersection_z.append(c[1])
    intersection_x.append(intersection[0][0])
    intersection_y.append(0)
    intersection_z.append(intersection[0][1])
    # print(frame, get_volume(intersection))
    intlist.append(get_volume(intersection))
    ax.plot(intersection_x,intersection_z,intersection_y)
    
    
    
    # transverse_ch = convex_hull(transverse)
    # transverse_x = []
    # transverse_y = []
    # transverse_z = []
    # for c in transverse_ch:
    #     transverse_x.append(c[0])
    #     transverse_y.append(0)
    #     transverse_z.append(c[1])
    # transverse_x.append(transverse_ch[0][0])
    # transverse_y.append(0)
    # transverse_z.append(transverse_ch[0][1])
    # # print(frame, get_volume(transverse_ch))
    # trlist.append(get_volume(transverse_ch))
    # ax.plot(transverse_x,transverse_z,transverse_y)
    
    # sagittal_ch = convex_hull(sagittal)
    # sagittal_x = []
    # sagittal_y = []
    # sagittal_z = []
    # for c in sagittal_ch:
    #     sagittal_x.append(0)
    #     sagittal_y.append(-c[0])
    #     sagittal_z.append(c[1])
    # sagittal_x.append(0)
    # sagittal_y.append(-sagittal_ch[0][0])
    # sagittal_z.append(sagittal_ch[0][1])
    # salist.append(get_volume(sagittal_ch))
    # ax.plot(sagittal_x,sagittal_z,sagittal_y)
    
    
    # coronal_ch = convex_hull(coronal)
    # coronal_x = []
    # coronal_y = []
    # coronal_z = []
    # for c in coronal_ch:
    #     coronal_x.append(c[0])
    #     coronal_y.append(-c[1])
    #     coronal_z.append(0)
    # coronal_x.append(coronal_ch[0][0])
    # coronal_y.append(-coronal_ch[0][1])
    # coronal_z.append(0)
    # colist.append(get_volume(coronal_ch))
    # ax.plot(coronal_x,coronal_z,coronal_y)
    

    ax.plot(leftleg[0], leftleg[1], leftleg[2])
    ax.plot(rightleg[0], rightleg[1], rightleg[2])
    ax.plot(leftarm[0], leftarm[1], leftarm[2])
    ax.plot(rightarm[0], rightarm[1], rightarm[2])


    ax.plot(spine[0], spine[1], spine[2])
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(-0.2, 0.2)
    ax.set_zlim(-0.5, 0.5)
    plt.draw()
    plt.pause(0.0001)
    ax.clear()

intdf = pd.DataFrame(intlist, columns=['area'])
intdf.to_csv("rotatelegtorsolegsseo.csv")
# trdf = pd.DataFrame(trlist, columns=['area'])
# trdf.to_csv("29upsseo.csv")
# sadf = pd.DataFrame(salist, columns=['area'])
# sadf.to_csv("29torsosaeun.csv")
# codf = pd.DataFrame(colist, columns=['area'])
# codf.to_csv("29torsocoeun.csv")