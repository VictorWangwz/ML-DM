import numpy as np
from sklearn.cluster import KMeans
from kmeans import Kmeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour

        """

        H, W, D = img.shape
        # model = KMeans(n_clusters=2**self.b, n_init=3)
        model=Kmeans(k=2**self.b)
        X=np.reshape(img,(H*W,3))
        model.fit(X)    
        y=model.predict(X)
        print(y.shape)
        # self.y=y
        # self.center=model.means
        # Reshape 2D-matrix to 3D-img
        # quantized_img = img
        # X=np.reshape(img,(H*W,3))
        # model.fit(X)
        # y=model.predict(X)
        # m=y.shape
        # print(m)
        # quantized_img=y
        
        self.colours = np.zeros((2**self.b,3),dtype='uint8')
                    # ,dtype='uint8')      
        for i in range(2**self.b):
                    # img[i, :] = quantized_img[i]
            self.colours[i, :] = model.means[i, :]
        img=np.zeros((H*W),dtype='uint8')
        for i in range(H*W):
            img[i]=y[i]
        img=np.reshape(img,(H,W))
        quantized_img=img


        # TODO: fill in code here
        # raise NotImplementedError()

        return quantized_img

      
    def dequantize(self, quantized_img):
        H, W= quantized_img.shape
        img=np.zeros((H,W,3),dtype='uint8')
        for i in range(H):
            for j in range(W):
                # print(quantized_img[i,j])
                img[i,j, :] = self.colours[quantized_img[i,j], :]
                    # img[i, :] = quantized_img[i]
            
        # img=np.reshape(img, (H, W, 3))
        # TODO: fill in the values of `img` here
        # raise NotImplementedError()

        return img
