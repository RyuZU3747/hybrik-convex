# hybrik-convex

convexhull.py의 115번째 줄의 file 변수로 hybrik의 pk파일을 불러와서, 대상의 움직임에 따른 convex hull을 구하고 면적 등을 계산합니다. 상체, 하체, 전신 등 지정한 parts의 convex hull끼리 겹치는 넓이도 구할 수 있습니다. sutherland_hodgman_clip 함수를 확인하시면 됩니다.
parts가 29개인 모델을 기준으로 작성되었습니다. 원하는 좌표축(x,y,z)에 임의로 projection 하여 구할 수 있습니다. pandas DataFrame을 사용해 csv 파일로 출력하여, plotting.py를 통해 그래프를 출력할 수 있습니다. 사용할 땐 두 환자를 비교하는 용도로 사용했습니다.
