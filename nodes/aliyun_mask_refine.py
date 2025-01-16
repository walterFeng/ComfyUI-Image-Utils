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

from .common_utils import pil2tensor, check_shape, image_to_mask


def refine_mask(url, url_mask):
    access_key_id = None
    access_key_secret = None
    try:
        this_folder = os.path.dirname(os.path.abspath(__file__))
        init_file = os.path.join(this_folder, '../aliyun.ini')
        config_reader = configparser.ConfigParser()
        config_reader.read(init_file, encoding='utf-8')
        access_key_id = config_reader.get('aliyun access', 'access_key_id')
        access_key_secret = config_reader.get('aliyun access', 'access_key_secret')
    except Exception as error:
        print(error)
    config = Config(
        access_key_id=access_key_id if access_key_id else os.environ.get('ACCESS_KEY_ID'),
        access_key_secret=access_key_secret if access_key_secret else os.environ.get('ACCESS_KEY_SECRET'),
        endpoint='imageseg.cn-shanghai.aliyuncs.com',
        region_id='cn-shanghai'
    )
    img_url = load_image(url)
    img_mask_url = load_image(url_mask)
    refine_mask_request = RefineMaskAdvanceRequest()
    refine_mask_request.image_urlobject = img_url
    refine_mask_request.mask_image_urlobject = img_mask_url
    runtime = RuntimeOptions()
    mask = None
    masked_url = url_mask
    try:
        client = Client(config)
        response = client.refine_mask_advance(refine_mask_request, runtime)
        print(response.body)
        data = response.body
        masked_url = data.data.elements[0].image_url
        mask = load_image(masked_url)
    except Exception as error:
        print(error)

    return Image.open(img_url), Image.open(mask if mask else img_mask_url), masked_url


def try_url_open(url):
    error_count = 0
    img_io = None
    while True:
        try:
            img_io = io.BytesIO(urlopen(url).read())
            break
        except Exception as e:
            error_count += 1
            print(e)
            if error_count >= 5:
                break
    return img_io


def load_image(image_source):
    if image_source.startswith('http'):
        print(image_source)
        img = try_url_open(image_source)
    else:
        file_obj = io.open(image_source, mode="rb")
        img = io.BytesIO(file_obj.read())
    return img


class RefineMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": True}),
                "mask_url": ("STRING", {"multiline": True}),
                "append_query": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "masked_url")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, image_url, mask_url, append_query):
        image_url = image_url + ("?" if image_url.find("?") == -1 else "&") + append_query
        mask_url = mask_url + ("?" if mask_url.find("?") == -1 else "&") + append_query
        image_loaded, mask_loaded, masked_url = refine_mask(image_url, mask_url)
        image, _ = pil2tensor(image_loaded)
        transform = transforms.ToTensor()
        mask = image_to_mask(check_shape(transform(mask_loaded)).unsqueeze(0), 'red')
        return image, mask, masked_url


if __name__ == "__main__":
    print("main")
    image_url1 = 'https://viapi-test-bj.oss-cn-beijing.aliyuncs.com/viapi-3.0domepic/imageseg/RefineMask/RefineMask1.jpg'
    mask_url1 = 'https://viapi-test-bj.oss-cn-beijing.aliyuncs.com/viapi-3.0domepic/imageseg/RefineMask/RefineMask6.jpg'
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
