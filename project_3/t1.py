import cv2
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact, Layout, GridspecLayout
import pickle

font = cv2.FONT_HERSHEY_SIMPLEX
width_img = 1280
height_img = 960

hue = widgets.IntRangeSlider(min=0, max=179, step=1, value=[0, 179], layout=Layout(width='100%'), description='H')
sat = widgets.IntRangeSlider(min=0, max=255, step=1, value=[0, 255], layout=Layout(width='100%'), description='S')
val = widgets.IntRangeSlider(min=0, max=255, step=1, value=[0, 255], layout=Layout(width='100%'), description='V')
x = widgets.IntSlider(min=0, max=width_img, step=1, value=100, layout=Layout(width='100%'), description='x')
y = widgets.IntSlider(min=0, max=height_img, step=1, value=100, layout=Layout(width='100%'), description='y')
w = widgets.IntSlider(min=0, max=width_img, step=1, value=300, layout=Layout(width='100%'), description='w')
h = widgets.IntSlider(min=0, max=height_img, step=1, value=300, layout=Layout(width='100%'), description='h')

stop_button = widgets.ToggleButton(description='Stop', disabled=False)
output = widgets.HTML(value='abc', description='Train:', disabled=False)
ID_widget = widgets.Text(description='ID_name')
train_button = widgets.ToggleButton(description='Train', disabled=False)
reset_button = widgets.ToggleButton(description='Reset', disabled=False)
add_layer_button = widgets.ToggleButton(description='Add_layer', disabled=False)
list_layers = widgets.HTML(value='0', description='Layers:', disabled=False)

grid0 = widgets.GridBox([stop_button, output], layout=widgets.Layout(grid_template_columns='repeat(2, 200px)'))
grid2 = widgets.GridBox([ID_widget, list_layers, train_button])
                        
grid = GridspecLayout(4, 3, height='130px', width='1300px')
grid[0,0], grid[1,0], grid[2,0]  = hue, sat, val
grid[0,1], grid[1,1], grid[2,1], grid[3,1] = x, y, w, h
grid[0,2], grid[1,2], grid[2,2] = add_layer_button, reset_button, list_layers

children = [grid0, grid, grid2]
tab = widgets.Tab()
tab.children = children
tab.titles = ['Main', 'Filters', 'Train']

def preprocess(fr):
    mask = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (5,5), 2)
    mask = cv2.Canny(mask, 30, 255, apertureSize=5)
    return mask

def contours(fr_m, fr):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(fr_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100_000:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(fr, cnt, -1, (0,255,0), 5)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def reorder(my_points):
    my_points = my_points.reshape((4,2))
    my_points_new = np.zeros((4,1,2), np.int32)
    add = my_points.sum(1)

    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]

    diff = np.diff(my_points, axis=1)

    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]

    return my_points_new

def get_warp(img, approx):
    biggest = reorder(approx)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width_img, height_img))
    img_cropped = cv2.resize(img_output, (width_img, height_img))
    image_grayworld = (img_cropped * (img_cropped.mean() / img_cropped.mean(axis=(0, 1)))).astype(np.uint8)
    return image_grayworld

def filters_online(fr, hue, sat, val, x, y, w, h):
    lower = np.array([hue.value[0], sat.value[0], val.value[0]])
    upper = np.array([hue.value[1], sat.value[1], val.value[1]])
    mask = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(mask, lower, upper)
    cv2.rectangle(mask, (x.value, y.value), (x.value+w.value, y.value+h.value), (255, 0, 0), 10)
    return mask

def text_detection(frame, img, contours_data, df, rectangles_sum, train, prediction): 
    tensor = np.empty((1,4))
    for contour in contours_data[:,:]:
        x, y, w, h = contour[0], contour[2], contour[1], contour[3]
        cv2.rectangle(img, (x, y), (w, h), (255, 255, 0), 6)

    if rectangles_sum.value == 0 and train.value == 0:
        st = 'Detecting...'
        color = (0, 0, 255)
    elif train.value == 1:
        st = 'Learning...'
        color = (255, 0, 0)
    else:   
        color = (0, 255, 0)
        st = df['Name'][df['ID'] == prediction.value].iloc[0]

    cv2.putText(frame, st, (7, 70), font, 3, color, 7, cv2.LINE_AA)
















