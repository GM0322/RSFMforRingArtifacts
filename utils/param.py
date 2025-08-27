import argparse
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser(description='Ring Artifact removing code')

    parser.add_argument('--train_path', type=str,default=r'D:\code\RingArtifacts\data\train\input')
    parser.add_argument('--val_path',type=str, default=r'D:\code\RingArtifacts\data\val\input')
    parser.add_argument('--log_path',type=str,default='log')
    parser.add_argument('--roi',type=tuple,default=(1,2,4))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=32)

    # CT scan param
    sod = 500.0
    odd = 500.0
    bins = 600
    size = 512
    views = 720
    cellSize = 1.1
    halfDet = cellSize * bins / 2.0
    length = np.sqrt((sod + odd) ** 2 + halfDet ** 2)
    pixelSize = sod * halfDet / (length * size).astype(np.float32).tolist()
    parser.add_argument('--sod',type=float,default=sod)
    parser.add_argument('--odd',type=float,default=odd)
    parser.add_argument('--bins',type=int,default=bins)
    parser.add_argument('--size',type=int,default=size)
    parser.add_argument('--views',type=int,default=views)
    parser.add_argument('--cellsize',type=float,default=cellSize)
    parser.add_argument('--pixelsize',type=float,default=pixelSize)
    parser.add_argument('--noise',type=float,default=0.2)

    return parser.parse_args()
