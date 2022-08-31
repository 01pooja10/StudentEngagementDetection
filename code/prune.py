def Prune(model1):
    m = copy.deepcopy(model1)
    m = m.cuda()
    #print(m.named_modules())
    for name, module in m.named_modules():
        if isinstance(module, nn.Conv3d):
            prune.l1_unstructured(module, name = 'weight', amount = 0.4)
            prune.remove(module, 'weight')
    return m
