import numpy as np
import astra
import os
from tqdm import tqdm
from utils import param

class G2L():
    def __init__(self,args):
        fSod = args.sod
        fOdd = args.odd
        nBins = args.bins
        nSize = args.size
        nViews = args.views
        fCellSize = args.cellsize
        fPixelSize = args.pixelsize
        self.args = args
        self.vol_geom = astra.create_vol_geom(nSize,nSize,-nSize/2.0*fPixelSize,nSize/2.0*fPixelSize,
                                              -nSize/2.0*fPixelSize,nSize/2.0*fPixelSize)
        self.proj_geom = astra.create_proj_geom('fanflat', fCellSize, nBins,np.linspace(0, 2 * np.pi, nViews, False), fSod, fOdd)
        self.proj_id = astra.create_projector('cuda',self.proj_geom,self.vol_geom)
        self.sin_id, sinogram = astra.create_sino(np.zeros((nSize,nSize),dtype=np.float32), self.proj_id)

    def discrete_gaussian(self, size, r=1,sigma=1):
        """在{0,1,2}上生成高斯式分布样本

        参数：
        size: 样本数量
        peak: 峰值位置（应为1）
        sigma: 聚集程度(0-1)，越小越集中
        """
        # 计算未归一化权重
        idx = np.array([i for i in range(2*r+1)])
        raw_weights = np.exp(-(1.0*idx - r) ** 2 / (2 * sigma ** 2)).astype(np.float32)
        norm_weights = raw_weights / np.sum(raw_weights)

        return np.random.choice(idx, size=size, p=norm_weights)

    def getMutilChannal(self,x,roi):
        # 1x1,max,mean,min,3x3,5x5,7x7
        roi_max =roi[-1]
        idx = np.array([i for i in range(x.shape[-1])],dtype=np.int32)
        o = []
        for i in range(-roi_max,roi_max+1):
            id = (idx+i)%x.shape[-1]
            temp = x[...,id]
            o.append(temp)
        o = np.array(o)
        max_channel = o[roi_max-2:roi_max+3,...].max(axis=0,keepdims=True)
        min_channel = o[roi_max-2:roi_max+3,...].min(axis=0,keepdims=True)
        mean_channel = o[roi_max-2:roi_max+3,...].mean(axis=0,keepdims=True)
        res = np.concatenate((x[None,...],max_channel,mean_channel,min_channel),axis=0)
        for r in roi:
            temp = o[roi_max-r:roi_max+r+1,...] # 3x512x512
            # channel_indices = np.random.randint(0, temp.shape[0], size=(temp.shape[1], temp.shape[2]))   # 0,1,2
            channel_indices = self.discrete_gaussian(size=(temp.shape[1], temp.shape[2]),r=r,sigma=r/2)   # 0,1,2
            x_indices, y_indices = np.mgrid[0:temp.shape[1], 0:temp.shape[2]]
            result = temp[channel_indices, x_indices, y_indices][None,...]
            res = np.concatenate((res,result),axis=0)
        return res


    def Recon(self,p):
        rec_id = astra.data2d.create('-vol', self.vol_geom)
        o = []
        for i in range(p.shape[0]):
            rec_fbp = None
            astra.data2d.store(self.sin_id, p[i,...])
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = self.sin_id
            cfg['option'] = {'FilterType': 'Ram-Lak'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            rec_fbp = astra.data2d.get(rec_id)
            astra.algorithm.delete(alg_id)
            astra.projector.delete(rec_id)
            o.append(rec_fbp)
        return np.array(o)

    def proj2image(self,x):
        x = self.getMutilChannal(x,self.args.roi)
        res = self.Recon(x)
        return res

    def pfile2ifile(self,path):
        save_path = path
        path =  path +'/../proj'
        files = os.listdir(path)
        loop = tqdm(enumerate(files),total=len(files))
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)
        for step, file in loop:
            x = np.fromfile(path+'/'+file,dtype=np.float32).reshape(self.args.views,self.args.bins)
            res = self.proj2image(x)
            res.tofile(save_path+'/'+file)

if __name__ == "__main__":
    astra.set_gpu_index(0)
    args = param.getArgs()
    gl = G2L(args)
    gl.pfile2ifile(args.train_path)
    gl.pfile2ifile(args.val_path)