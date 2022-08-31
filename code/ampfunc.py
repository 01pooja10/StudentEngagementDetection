from .plotting import plot

def AMP_Train(mod, loader, opt, epochs, batch, nc):
    batch = batch-1
    losses = []
    bce = nn.BCEWithLogitsLoss()
    #m = Prune(mod)
    m = mod
    m.cuda()
    m.train()
    opt_level = 'O2'
    opt.zero_grad()
    #m, opt = amp.initialize(m, opt, opt_level=opt_level)
    scaler = GradScaler(enabled=True)
    

    for e in range(epochs):
        for n, (x,y) in enumerate(loader):
            #print(len(x))
            #print(x.size())
            a = x.float().to('cuda:0',non_blocking=True)
            b = y.float().squeeze().to('cuda:0',non_blocking=True)
            #print(a.size())
            
            with autocast():
                res = m(a)
                out = res.squeeze()
                #out = out.squeeze()
                #print(out.size())
                #print('result', out)
                #print(b.size())
                #print('before loss', b)
                loss = bce(out, b)
            
            scaler.scale(loss).backward()
            if (n+1) % 2 == 0 or (n+1) == len(dl):
                #scaler.step()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            

        losses.append(loss.item())

        print('epoch: ', str(e+1), 'Loss: ', str(loss.item()))

    plot(losses)
    model_data = {'model': m.state_dict(),'optimizer': opt.state_dict(),'loss': losses}
    torch.save(model_data, 'net3d_5f.pth')
    print('done')
