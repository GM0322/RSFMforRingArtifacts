from model.network import RectifiedFlow,UnetAttnCondition
from model.UNet import AttnUnet
from utils.data import MultiChannelData
from utils.param import getArgs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


if __name__ == "__main__":
    args = getArgs()
    channel = 7
    train_data = MultiChannelData(args.train_path,args.size,channel)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,
                              shuffle=True,num_workers=4,pin_memory=True,pin_memory_device=args.device)

    val_data = MultiChannelData(args.val_path, args.size, channel)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    model = AttnUnet(in_channels=7,out_channels=1,base_features=16).to(args.device)
    ReFlow = RectifiedFlow(model=model, num_steps=100)
    optimizer = torch.optim.Adam(ReFlow.model.parameters(), lr=1e-4)
    loss_function = torch.nn.MSELoss(reduction='sum')
    loss_curve = []
    for epoch in range(args.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (x, y, file) in loop:
            optimizer.zero_grad()
            # z0 = x.to(args.device).detach().clone()
            z1 = y.to(args.device).detach().clone()
            z0 = x.to(args.device).detach().clone()
            z_t, t, target = ReFlow.get_train_tuple(z0=z0, z1=z1)
            pred = ReFlow.model(z_t, t)
            loss = loss_function(pred, target)
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())  ## to store the loss curve
            loop.set_description(f'Epoch [{epoch}/{args.epochs}],'
                                 f'[{step}/{len(train_loader)}],'
                                 f'loss:{loss.item():.6f}')

        traj, out = ReFlow.sample_ode(z0=z0)
        out.data.cpu().numpy().tofile('res.raw')
        z0.data.cpu().numpy().tofile('z0.raw')