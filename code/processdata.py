class Daisee(Dataset):
    def __init__(self, imgs, labels, batch=1):
        
        super(Daisee, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.bs = batch

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]      
        #print(img.size())  
        img = torch.permute(img, (3, 1, 2, 0))
        #img = torch.permute(img, (2, 0, 1))
        lab = self.labels[idx]
        lab = torch.as_tensor(lab.float())
        return img, lab
