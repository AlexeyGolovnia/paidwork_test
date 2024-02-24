import pickle
import numpy as np
import torch
import torch.nn as nn
from multiprocessing.shared_memory import SharedMemory
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class Model_ID(nn.Module):
    def __init__(self, in_layers, out_layers):
        super(Model_ID, self).__init__()
        self.start = nn.Sequential(
            nn.Linear(in_layers, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.hidden = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(512, out_layers)

    def forward(self, x):
        x = self.start(x)
        x = self.hidden(x)
        x = self.out(x)
        return x

def data_prepare(itemlist):
    s = 0 
    for i in itemlist:
        if s < i[0].shape[0]:
            s = i[0].shape[0]
            
    ret_input = np.empty((1, s * 4))
    ret_target = []
            
    for i in itemlist:
        t = np.pad(i[0], [(0, s - i[0].shape[0]),(0, 0)], mode='constant')
        ret_input = np.vstack((ret_input, t.T.reshape(1, -1)))
        ret_target.append(i[1])

    return ret_input[1:,:], ret_target, s

def model_detect(rectangles_sum, prediction):
    rectangles_memory = SharedMemory('RectanglesMemory')
    contours_memory = SharedMemory('ContoursMemory')
    with open('l2.pkl', 'rb') as f:
        l2 = pickle.load(f)
        
    pr = data_prepare(l2)
    model = Model_ID(pr[0].shape[1], pr[1][-1]+1)
    model.load_state_dict(torch.load('temp_model_0.pth', map_location=torch.device('cpu')))
    m = int(pr[0].shape[1] / 4)

    while True:
        if rectangles_sum.value > 0:
            contours_data = np.ndarray((rectangles_sum.value, 4), dtype=np.int32, buffer=contours_memory.buf)
            t = contours_data[:m,:].astype(int)
            t = np.pad(t, [(0, m - t.shape[0]),(0, 0)], mode='constant')
            t = t.T.reshape(1, -1)
            x11 = torch.tensor(t).float().to(device)
            p = model(x11)
            prediction.value = p.argmax(1)

            
            




        
    # while True:
    #    if aaa.value % 2 == 0:
    #        bbb.value = 88
    #    else:
    #        bbb.value = 99








































    


