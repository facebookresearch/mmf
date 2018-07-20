from PIL import Image
from PIL import ImageOps
import os
import json
from multiprocessing.dummy import Pool as ThreadPool

split = 'val2014'
image_paths = []


def mirror_image(image_path):
    img = Image.open(image_path)
    mirror_img = ImageOps.mirror(img)
    image_name = image_path.split('/')[-1]
    fh = '/private/home/nvivek/data/' + split
    fh = os.path.join(fh, image_name)
    mirror_img.save(fh, "JPEG")


with open('/datasets01/COCO/060817/annotations/instances_val2014.json') as f:
    data = json.load(f)
    for item in data['images']:
        image_id = int(item['id'])
        filepath = os.path.join('/datasets01/COCO/060817/val2014/',
                                item['file_name'])
        image_paths.append(filepath)

pool = ThreadPool(10)
results = pool.map(mirror_image, image_paths)
