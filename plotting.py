import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import csv

eun = pd.read_csv("29torsotreun.csv")
seo = pd.read_csv("29torsotrseo.csv")
# eun2 = pd.read_csv("29torsotreun.csv")
# seo2 = pd.read_csv("29torsotrseo.csv")

x1 = eun['area'].tolist() # 900~1100 1840~1940 2600~2730
x2 = seo['area'].tolist() # 180~250 400~460 630~730 870~950 1100~1180 1320~ 1500~ 1730~ 1960~ 2170~
# x12 = eun2['area'].tolist() # 900~1100 1840~1940 2600~2730
# x22 = seo2['area'].tolist() # 180~250 400~460 630~730 870~950 1100~1180 1320~ 1500~ 1730~ 1960~ 2170~

# x1 = [x1[i] + x12[i] for i in range(len(x1))]
# x2 = [x2[i] + x22[i] for i in range(len(x2))]

fig = plt.figure()
eunp = fig.add_subplot(211)
eunp.plot(x1)
# eunp.plot(x1[900:1100])
eunp.set_title("Patient")
plt.ylim(0, 0.004)
seop = fig.add_subplot(212)
seop.plot(x2)
seop.set_title("Healthy")
plt.ylim(0, 0.004)
plt.show()

# video = cv2.VideoCapture("res_2d_seo.mp4")

# fm = 0

# while video.isOpened():
#     check, frame = video.read()
#     if not check:
#         print("Frame이 끝났습니다.")
#         break
#     print(fm)
#     fm += 1
#     frame = cv2.resize(frame, (500,500))
#     cv2.imshow("video",frame)
#     if cv2.waitKey(25) == ord('q'):
#         print("동영상 종료")
#         break

# video.release()
# cv2.destroyAllWindows()
