import pathlib
from pathlib import Path
import PIL
from PIL import Image
import os

def make_square(im, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (((size - x) // 2), ((size - y) // 2)))
    return new_im.resize((768, 768), PIL.Image.BICUBIC)

datapath = "C:/Users/caleb/COMP 562/Final Project/KDEF_S"
emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

for emotion in emotions:
	for root, dirs, files in os.walk(os.path.join(os.path.abspath(datapath), emotion)):
		for file in files:
			path = os.path.join(root, file)
			test_image = Image.open(path)
			new_image = make_square(test_image)
			outpath = os.path.join(os.path.abspath(datapath), emotion + "_resized", os.path.splitext(file)[0] + ".png")
			new_image.save(outpath)
			print(path, "=>", outpath)