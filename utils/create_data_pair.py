import numpy as np
import astra
import os
from tqdm import tqdm
from utils import param

class DataGenrate():
    def __init__(self,args):
        fSod = args.sod
        fOdd = args.odd
        nBins = args.bins
        nSize = args.size
        nViews = args.views
        fCellSize = args.cellsize
        fPixelSize = args.pixelsize

        self.vol_geom = astra.create_vol_geom(nSize,nSize,-nSize/2.0*fPixelSize,nSize/2.0*fPixelSize,
                                              -nSize/2.0*fPixelSize,nSize/2.0*fPixelSize)
        self.proj_geom = astra.create_proj_geom('fanflat', fCellSize, nBins,np.linspace(0, 2 * np.pi, nViews, False), fSod, fOdd)
        self.proj_id = astra.create_projector('cuda',self.proj_geom,self.vol_geom)

    def projection(self,ImageData,isReon=False):
        sin_id, sinogram = astra.create_sino(ImageData,self.proj_id)
        noisy_sinogram = sinogram+args.noise*np.random.randn(1,sinogram.shape[1]).astype(np.float32)
        noisy_sinogram[noisy_sinogram<0] = 0
        rec_fbp_noisy = None
        rec_sirt_noisy = None
        if isReon == True:
            rec_id = astra.data2d.create('-vol',self.vol_geom)
            astra.data2d.store(sin_id,noisy_sinogram)
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sin_id
            # alg_id = astra.algorithm.create(cfg)
            # astra.algorithm.run(alg_id,100)
            # astra.algorithm.delete(alg_id)
            # rec_sirt_noisy = astra.data2d.get(rec_id)
            cfg['type'] = 'FBP_CUDA'
            cfg['option'] = {'FilterType': 'shepp-logan'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            rec_fbp_noisy= astra.data2d.get(rec_id)
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sin_id)
            astra.projector.delete(rec_id)
        return noisy_sinogram,rec_fbp_noisy,rec_sirt_noisy

    def generate(self,path):
        if(os.path.isdir(path+'/train/proj') == False):
            os.makedirs(path+'/train/proj',exist_ok=True)
        if(os.path.isdir(path+'/train/fbp') == False):
            os.makedirs(path+'/train/fbp',exist_ok=True)
        if(os.path.isdir(path+'/train/sirt') == False):
            os.makedirs(path+'/train/sirt',exist_ok=True)
        if(os.path.isdir(path+'/val/proj') == False):
            os.makedirs(path+'/val/proj',exist_ok=True)
        if(os.path.isdir(path+'/val/fbp') == False):
            os.makedirs(path+'/val/fbp',exist_ok=True)
        if(os.path.isdir(path+'/val/sirt') == False):
            os.makedirs(path+'/val/sirt',exist_ok=True)
        print('---------------------------start generate train data-----------------------------------------')
        files = os.listdir(path+'/train/label')
        for step,file in tqdm(enumerate(files),total=len(files)):
            img = np.fromfile(path+'/train/label/'+file,dtype=np.float32).reshape(512,512)
            noisy_sinogram, rec_fbp_noisy, rec_sirt_noisy = self.projection(img,True)
            if noisy_sinogram is not None:
                noisy_sinogram.tofile(path + '/train/proj/' + file)
                rec_fbp_noisy.tofile(path+'/train/fbp/'+file)
                # rec_sirt_noisy.tofile(path+'/train/sirt/'+file)
        print('---------------------------train data generate finish-----------------------------------------')
        print('---------------------------start generate val data-------------------------------------------')
        files = os.listdir(path+'/val/label')
        for step,file in tqdm(enumerate(files),total=len(files)):
            img = np.fromfile(path+'/val/label/'+file,dtype=np.float32).reshape(512,512)
            noisy_sinogram, rec_fbp_noisy, rec_sirt_noisy = self.projection(img,True)
            if noisy_sinogram is not None:
                noisy_sinogram.tofile(path+'/val/proj/'+file)
                rec_fbp_noisy.tofile(path+'/val/fbp/'+file)
                # rec_sirt_noisy.tofile(path+'/val/sirt/'+file)
        print('---------------------------validation  data generate finish------------------------------------')


if __name__ == '__main__':
    astra.set_gpu_index(1)
    args = param.getArgs()
    dg = DataGenrate(args)
    spath = r'D:\code\RingArtifacts\data'
    dg.generate(spath)
