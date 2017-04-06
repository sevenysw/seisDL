import scipy.io as sio
import numpy as np
    
class load_data():
# load data
    def __init__(self):
        data = sio.loadmat('test.mat')
        self.images = np.transpose(data['Yf'])
        self.labels = np.transpose(data['Xf'])
		
        #data_train = sio.loadmat('train.mat')
        #self.train_images = np.transpose(data_train['Yt'])
        #self.train_labels = np.transpose(data_train['Xt'])
        self.ind = 0
# load train data
    def next_batch(self,n):
        t = self.ind
        self.ind = self.ind + n
        t1 = self.train_images[t:t+n,:,:]
        t2 = self.train_labels[t:t+n,:,:]
        return t1[..., np.newaxis], t2[..., np.newaxis]
# test data
    def test_data(self):
        i = 10
        j = 10
        data = sio.loadmat('data1.mat')
        tmp = data['d']
        t1 = tmp[i:i+28,j:j+27:2]
        t2 = tmp[i:i+28,j+1:j+28:2]
        return t1[np.newaxis,...,np.newaxis],t2[np.newaxis,...,np.newaxis]
                

if __name__ == "__main__":
    sd = load_data()
    d = sd.test_data()
    print(d)
