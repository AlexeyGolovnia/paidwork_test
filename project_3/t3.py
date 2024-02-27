import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from multiprocessing.shared_memory import SharedMemory
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class MyDataset(Dataset):
    def __init__(self, X, y, sh):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(np.array(y)).type(torch.LongTensor)

    def __getitem__(self, x):
        return self.X[x].to(device), self.y[x].to(device)

    def __len__(self):
        return self.X.shape[0]

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

def model_detect(rectangles_sum, prediction, train, tmp2, ID_name):
    def train_batch(x, y, model, opt, loss_fn):
        model.train()
        prediction = model(x)
        batch_loss = loss_fn(prediction, y)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return batch_loss.item()
    
    rectangles_memory = SharedMemory('RectanglesMemory')
    contours_memory = SharedMemory('ContoursMemory')
    with open('data/l2.pkl', 'rb') as f:
        l2 = pickle.load(f)
    with open('data/df.pkl', 'rb') as f:
        df = pickle.load(f)
        
    pr = data_prepare(l2)
    model = Model_ID(pr[0].shape[1], pr[1][-1]+1)
    model.load_state_dict(torch.load('data/temp_model_0.pth', map_location=torch.device('cpu')))
    m = int(pr[0].shape[1] / 4)

    while True:
        if rectangles_sum.value > 0 and train.value == 0 and tmp2.value == 1:
            contours_data = np.ndarray((rectangles_sum.value, 4), dtype=np.int32, buffer=contours_memory.buf)
            t = contours_data[:m,:].astype(int)
            t = np.pad(t, [(0, m - t.shape[0]),(0, 0)], mode='constant')
            t = t.T.reshape(1, -1)
            x11 = torch.tensor(t).float().to(device)
            p = model(x11)
            prediction.value = p.argmax(1)
            tmp2.value = 0

        if train.value == 1 and rectangles_sum.value > 0:
            n = df['ID'].max()
            c = 0
            while c < 50:
                if tmp2.value == 1:
                    contours_data = np.ndarray((rectangles_sum.value, 4), dtype=np.int32, buffer=contours_memory.buf)
                    l2.append((contours_data[:,:].astype(int), n+1))
                    c += 1
                    tmp2.value = 0

            pr = data_prepare(l2)
            model.start = nn.Sequential(nn.Linear(pr[0].shape[1], 1024), nn.ReLU(), nn.Dropout(0.5))
            model.out = nn.Linear(512, pr[1][-1]+1)
            train_data = MyDataset(pr[0], pr[1], pr[0].shape[1])
            train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr= 1e-5)

             
            while True:
                train_epoch_losses = []
                for ix, batch in enumerate(train_dataloader):
                    x, y = batch
                    batch_loss = train_batch(x, y, model, optimizer, loss_fn)
                    train_epoch_losses.append(batch_loss)
                train_epoch_loss = np.array(train_epoch_losses).mean()
                if train_epoch_loss < 0.5:
                    break

            m = int(pr[0].shape[1] / 4)
            torch.save(model.state_dict(), 'data/temp_model_0.pth')
            df.loc[len(df.index)] = [n+1, ID_name.value.decode('utf-8')]
            with open('data/df.pkl', 'wb') as f:
                pickle.dump(df, f)
            with open('data/l2.pkl', 'wb') as f:
                pickle.dump(l2, f)
            
            train.value = 2










































    


