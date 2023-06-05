from pathlib import Path
from PIL import Image

def make_square(im, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (((size - x) // 2), ((size - y) // 2)))
    return new_im.resize((768, 768), Image.BICUBIC)

datapath = Path("C:/Users/caleb/COMP 562/Final Project/KDEF_S")
emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

for emotion in emotions:
    for path in datapath.glob(f"{emotion}/**/*.jpg"):
        image_path = Path(path)
        outpath = datapath / f"{emotion}_resized" / (image_path.stem + ".png")

        with Image.open(image_path) as test_image:
            new_image = make_square(test_image)
            new_image.save(outpath)

        print(path, "=>", outpath)
