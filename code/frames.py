def AllocateFrames(imgs):
	
	if args.frames == "1 frame":
		fs = np.zeros((1531,128,128,3))
		c = 0
		for i in range(len(imgs)):  
			fs[c,:,:,:] = imgs[i][5]
			c += 1
		fs = torch.FloatTensor(fs)
		fs1 = fs[:1384]
		return fs1

	elif args.frames == "3 frames":
		fs = np.zeros((1531,3,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(4,7):
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs3 = fs[:1384]
		return fs3

	elif args.frames == "5 frames":	
		fs = np.zeros((1531,5,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(3,8):
				#print('init', i,j)
				#print(c,d)
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs5 = fs[:1384]
		return fs5

	elif args.frames == "7 frames":
		fs = np.zeros((1531,7,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(3,10):
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs7 = fs[:1384]
		return fs7
