import configparser
import io
import os
from urllib.request import urlopen
import torchvision.transforms as transforms

from PIL import Image
from alibabacloud_imageseg20191230.client import Client
from alibabacloud_imageseg20191230.models import RefineMaskAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

from .common_utils import pil2tensor

def refine_mask(url, urlMask):
    access_key_id = None
    access_key_secret = None
    try:
        thisfolder = os.path.dirname(os.path.abspath(__file__))
        initfile = os.path.join(thisfolder, '../aliyun.ini')
        configReader = configparser.ConfigParser()
        configReader.read(initfile, encoding='utf-8')
        access_key_id = configReader.get('aliyun access', 'access_key_id')
        access_key_secret = configReader.get('aliyun access', 'access_key_secret')
    except Exception as error:
        print(error)
    config = Config(
        access_key_id=access_key_id if access_key_id else os.environ.get('ACCESS_KEY_ID'),
        access_key_secret=access_key_secret if access_key_secret else os.environ.get('ACCESS_KEY_SECRET'),
        endpoint='imageseg.cn-shanghai.aliyuncs.com',
        region_id='cn-shanghai'
    )
    imgUrl = load_image(url)
    imgMaskUrl = load_image(urlMask)
    refine_mask_request = RefineMaskAdvanceRequest()
    refine_mask_request.image_urlobject = imgUrl
    refine_mask_request.mask_image_urlobject = imgMaskUrl
    runtime = RuntimeOptions()
    mask = None
    mask_url = ''
    try:
        client = Client(config)
        response = client.refine_mask_advance(refine_mask_request, runtime)
        print(response.body)
        data = response.body
        mask_url = data.data.elements[0].image_url
        mask = load_image(mask_url)
    except Exception as error:
        print(error)

    return Image.open(imgUrl), Image.open(mask if mask else imgMaskUrl), mask_url


def tryUrlOpen(url):
    errorCount = 0
    imgIO = None
    while True:
        try:
            imgIO = io.BytesIO(urlopen(url).read())
            break
        except Exception as e:
            errorCount += 1
            print(e)
            if errorCount >= 5:
                break
    return imgIO


def load_image(image_source):
    if image_source.startswith('http'):
        print(image_source)
        img = tryUrlOpen(image_source)
    else:
        file_obj = io.open("data.txt", mode="rb")
        img = io.BytesIO(file_obj.read())
    return img


class RefineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": True}),
                "mask_url": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "masked_url")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image_url, mask_url):
        image_loaded, mask_loaded, masked_url = refine_mask(image_url, mask_url)
        image, _ = pil2tensor(image_loaded)
        transform = transforms.ToTensor()
        mask = transform(mask_loaded)
        return image, mask, masked_url


if __name__ == "__main__":
    print("main")
    image_url1 = 'http://oss-shared.oss-cn-beijing.aliyuncs.com/uploads/test-111-img-1226.jpg'
    mask_url1 = 'http://oss-shared.oss-cn-beijing.aliyuncs.com/uploads/test-111-img-1226-mask.png'
    image_loaded1, mask_loaded1, masked_url1 = refine_mask(image_url1, mask_url1)
    image1, _ = pil2tensor(image_loaded1)
    transform1 = transforms.ToTensor()
    mask1 = transform1(mask_loaded1)
    print(image1)
    # 将张量转换为PIL图像的函数
    toPIL = transforms.ToPILImage()
    # 将张量转换为PIL图像
    pic = toPIL(mask1)
    # 将PIL图像保存为JPEG文件
    pic.save('mask.jpg')
    print(mask1)
    print(masked_url1)
