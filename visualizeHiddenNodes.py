from PIL import Image
from nn import NeuralNet

img = Image.new('RGB', (255, 255), "black")  # create a new black image
pixels = img.load()  # create the pixel map

nn = NeuralNet(1, numEpochs=500)
hidden_layers = nn.get_hidden_layer()
for i in range(img.size[0]):  # for every pixel:
    for j in range(img.size[1]):

        pixels[i, j] = (i, j, 100)  # set the colour accordingly

img.show()