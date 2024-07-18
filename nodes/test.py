from PIL import Image

def calculate_brightness(image_path):
    img = Image.open(image_path).convert('RGBA')
    pixels = img.getdata()
    brightness = 0
    count = 0
    for pixel in pixels:
        if pixel[3] != 0:
            brightness += (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            count += 1

    if count != 0:
        average_brightness = brightness / count
    else:
        average_brightness = 0

    print(f'Average brightness (excluding transparent areas): {average_brightness}')
    return average_brightness


if __name__ == '__main__':
    #file = sys.argv[1:][0]
    file = "../heisetu.png"
    print("%s\t%s" % (file, calculate_brightness(file) / 255.0))