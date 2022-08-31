def plot(loss):
  plt.figure(figsize=(15,4))
  plt.title('Loss curve')
  plt.plot(loss)
  plt.savefig('loss.png')
