import scipy.io as sio
import numpy as np
    
class load_data():
# load data
    def __init__(self):
        data = sio.loadmat('../data/seismic.mat')
        datas = data['data']
        self.data_train = datas[0,1]/255
        self.data_test  = datas[0,0]/255		
        self.t1 = int(0)
        self.t2 = int(0)
# load train data
    def next_batch(self,n):
        ti = int(0)
        r = int(28)
        s = int(1)
        [n1,n2] = self.data_train.shape
        images = np.zeros([n,r,r/2])
        labels = np.zeros([n,r,r/2])
        for i in range(self.t1,n1-r+1,s):
            #print(i)
            for j in range(self.t2,n2-r+1,s):
                ti = ti + 1
                if (ti < n):
                    images[ti,:,:] = self.data_train[i:i+r,j:j+r:2]
                    #print(images.shape)
                    labels[ti,:,:] = self.data_train[i:i+r,j+1:j+r:2]
                    #print(i,j,labels.shape)
                else :
                    self.t1 = i
                    self.t2 = j
                    return images[..., np.newaxis], labels[..., np.newaxis]
            self.t2 = 0
            
# test data
    def test_data(self):
        i = 10
        j = 10
        tmp = self.data_test
        t1 = tmp[i:i+28,j:j+27:2]
        t2 = tmp[i:i+28,j+1:j+28:2]
        return t1[np.newaxis,...,np.newaxis],t2[np.newaxis,...,np.newaxis]
                

if __name__ == "__main__":
    sd = load_data()
    d1,d2 = sd.next_batch(400)
    print(d1.shape)
    d1,d2 = sd.next_batch(100)
    print(d2.shape)
