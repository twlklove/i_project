from PIL import Image, ImageFilter,ImageDraw, ImageFont
import random
def test_0():
    im = Image.open('E:/i_share/i_test/0.jfif')
    w, h = im.size
    print('Original image size: %sx%s' % (w, h))
    im.show()

    # 缩放到50%:
    im.thumbnail((w//2, h//2))
    print('Resize image to: %sx%s' % (w//2, h//2))
    im.show()
    #im.save('thumbnail.jpg', 'jpeg') # 把缩放后的图像用jpeg格式保存:

    # 应用模糊滤镜:
    im2 = im.filter(ImageFilter.BLUR)
    im2.show()
    #im2.save('blur.jpg', 'jpeg')

# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))

# 随机颜色:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def test_1():
    width = 60 * 4
    height = 60
    image = Image.new('RGB', (width, height), (0, 0, 255))
    image.show()
    # 创建Font对象:
    font = ImageFont.truetype('C:\Windows\Fonts\Arial.ttf', 36)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    for x in range(width):
        for y in range(height):
            draw.point((x, y), fill=rndColor())
    # 输出文字:
    for t in range(4):
        draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
    # 模糊:
    image = image.filter(ImageFilter.BLUR)
    image.show()
    #image.save('code.jpg', 'jpeg')

if __name__ == '__main__' :
    #test_0()
    test_1()