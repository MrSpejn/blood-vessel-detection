from PIL import Image
import glob, os

size = 876, 584

for infile in glob.glob("*.tif"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.save("_".join(file.split(".")) + ".jpg", "JPEG")