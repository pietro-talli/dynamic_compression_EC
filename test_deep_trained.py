import torch 
import compressai
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt




checkpoint = torch.load('../checkpoint_best_loss_more.pth.tar', map_location=torch.device('cpu'))
model = compressai.zoo.cheng2020_attn(quality=3, pretrained=False)
model.load_state_dict(checkpoint['state_dict'])
model.update()
model.eval()

x = Image.open("../dataset/images/0_0.png")

transform = torchvision.transforms.Compose([
    torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize((64, 128)),
    # convert from RGB to grayscale
    #torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ConvertImageDtype(torch.float32),
])

x = transform(x).unsqueeze(0)

with torch.no_grad():
    z =model.compress(x)
    print(len(z['strings'][0][0]))
    print(len(z['strings'][1][0]))
    x_hat = model.decompress(z['strings'], z['shape'])
    
plt.imshow(x_hat['x_hat'].squeeze().permute(1, 2, 0))
im_to_save = x_hat['x_hat'].squeeze().permute(1, 2, 0)
im_to_save = im_to_save.numpy()
plt.imsave('test.png', im_to_save)
#plt.show()

# compute PSNR

mse = torch.mean((x_hat['x_hat'] - x) ** 2)
psnr = -10 * torch.log10(mse)
print(f"PSNR: {psnr:.4} dB")
