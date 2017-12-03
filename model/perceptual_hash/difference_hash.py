# coding=utf-8


import fire
import sys, os
import pickle
from tqdm import tqdm

try:
    import PIL.Image
except ImportError:
    PIL = None

try:
    import wand.image
except ImportError:
    wand = None


"""
Difference Hashing (dHash)
- 将图片转为灰度图片
- 降采样为 9*9 的灰度图（或 17*17 的更大的 512 bit 的哈希）
- 计算行哈希（row hash）：对于每一行，从左侧移动到右侧，如果下一个灰度值大于或等于前一个，
  则输出 1，否则输出 0（每个 9 像素的行生成 8 bit 的输出）
- 计算列哈希（column hash）：与上面相同，从上移动到下
- 合并两个 64 bit 的值，形成 128 bit 的哈希
"""


def get_grays(image, width, height):
    """
    将图片转为灰度图片，并将图片缩小到 width*height，返回灰度像素值列表
    """
    if isinstance(image, (list, tuple)):
        if len(image) != width*height:
            raise ValueError('image sequence length ({}) not equal to width*height ({}).'.format(
                len(image), width*height
            ))
        return image

    if wand is None and PIL is None:
        raise ImportError('must have wand or Pillow/PIL installed to use dhash on images.')

    if wand is not None and isinstance(image, wand.image.Image):
        with image.clone() as small_image:
            small_image.type = 'grayscale'
            small_image.resize((width, height))
            blob = small_image.make_blob(format='RGB')
            return list(blob[::3])

    elif PIL is not None and isinstance(image, PIL.Image.Image):
        gray_image = image.convert('L')
        small_image = gray_image.resize((width, height), PIL.Image.ANTIALIAS)
        return list(small_image.getdata())

    else:
        raise ValueError('image must be a wand.image.Image or PIL.Image instance.')


def dhash_row_col(image, size=8):
    """
    计算给定图片的行和列的差异哈希（difference hashing），返回 (row_hash, col_hash) 的哈希值，每个值都是 size*size 比特的整型.
    """
    width = size + 1
    grays = get_grays(image, width, width)

    row_hash = 0
    col_hash = 0

    for y in range(size):
        for x in range(size):
            offset = y * width + x
            row_bit = grays[offset] < grays[offset + 1]
            row_hash = row_hash << 1 | row_bit

            col_bit = grays[offset] < grays[offset + width]
            col_hash = col_hash << 1 | col_bit

    return (row_hash, col_hash)


def dhash_int(image, size=8):
    """
    计算给定图片行、列差异哈希（difference hashing），返回 2*size*size 位大小的组合哈希值，row_hash 在高位，col_hash 在低位.
    """
    row_hash, col_hash = dhash_row_col(image, size=size)

    return row_hash << size * size | col_hash


def get_num_bits_different(hash1, hash2):
    """
    计算两个哈希值不同比特位的个数.
    """
    return bin(hash1 ^ hash2).count('1')


def format_bytes(row_hash, col_hash, size=8):
    """
    将 dhash 整型哈希值转为 size*size // 8 字节的二值字符串（row_hash 和 col_hash 拼接）.
    """
    bits_per_hash = size * size
    full_hash = row_hash << size * size | col_hash
    return full_hash.to_bytes(bits_per_hash // 4, 'big')


def format_hex(row_hash, col_hash, size=8):
    """
    将 dhash 整型哈希值转为 size*size // 2 十六进制位的十六进制字符串（row_hash 和 col_hash 拼接）.
    """
    hex_length = size*size // 4
    return '{0:0{2}x}{1:0{2}x}'.format(row_hash, col_hash, hex_length)


def format_matrix(hash_int, bits=None, size=8):
    """
    将 dhash 整型哈希值转为比特矩阵.
    """
    if bits is None:
        bits = ['0 ', '1 ']
    output = '{:0{}b}'.format(hash_int, size*size)
    output = output.translate({ord('0'): bits[0], ord('1'): bits[1]})
    width = size * len(bits[0])
    lines = [output[i:i+width] for i in range(0, size*width, width)]
    return '\n'.join(lines)


def format_grays(grays, size=8):
    """
    灰度值矩阵
    """
    width = size + 1
    lines = []
    for y in range(width):
        line = []
        for x in range(width):
            gray = grays[y * width + x]
            line.append(format(gray, '4'))
        lines.append(''.join(line))
    return '\n'.join(lines)


def force_pil():
    """
    如果 wand 和 Pillow/PIL 都安装了，强制使用 PIL
    """
    global wand
    if PIL is None:
        raise ValueError('Pillow/PIL library must be installed to use force_pil().')
    wand = None


def load_image(filename):
    """
    读入图片
    :param filename: 1 或 2 个图片文件名
    :return:
    """
    if wand is not None:
        return wand.image.Image(filename=filename)
    elif PIL is not None:
        return PIL.Image.open(filename)
    else:
        sys.stderr.write('You must have wand or Pillow/PIL installed to use the dhash command\n')
        sys.exit(1)


def test(filename=None, size=8, form='hex', pil=True):
    """
    测试
    计算两张图片 hamming distance
    :param filename:
    :param size:
    :param format:
    :param pil:
    :return:
    """
    if filename is None:
        filename = ['../data/test3.png', '../data/test7.png']

    if pil:
        try:
            force_pil()
        except ValueError:
            sys.stderr.write('You must have Pillow/PIL installed to use --pil\n')
            sys.exit(1)

    if len(filename) == 1:
        image = load_image(filename[0])
        if form == 'grays':
            grays = get_grays(image, size + 1, size + 1)
            print(format_grays(grays, size=size))
        else:
            row_hash, col_hash = dhash_row_col(image, size=size)
            if form == 'hex':
                print(format_hex(row_hash, col_hash, size))
            elif form == 'decimal':
                print(row_hash, col_hash)
            else:
                bits = ['. ', '* ']
                print(format_matrix(row_hash, bits=bits, size=size))
                print()
                print(format_matrix(col_hash, bits=bits, size=size))

    elif len(filename) == 2:
        image1 = load_image(filename[0])
        image2 = load_image(filename[1])
        hash1 = dhash_int(image1, size)
        hash2 = dhash_int(image2, size)
        num_bits_different = get_num_bits_different(hash1, hash2)
        print('{} {} out of {} ({:.1f}%)'.format(
            num_bits_different,
            'bit differs' if num_bits_different == 1 else 'bits differ',
            size * size * 2,
            100 * num_bits_different / (size * size * 2)
        ))

    else:
        sys.stderr.write('You must specify one or two filenames')


if __name__ == '__main__':
    # 多个函数时，使用 fire 需要指定函数名
    # fire.Fire()
    DATA_ROOT = '/data/wangyuan/koubei_image/data'
    image_path_pickle_file = os.path.join(DATA_ROOT, 'image_path.pkl')
    if not os.path.exists(image_path_pickle_file):
        print('image path does not exist.')
    else:
        with open(image_path_pickle_file, 'rb') as f:
            image_path = pickle.load(f)

        image_dhash = {}
        image_dhash_pickle_file = os.path.join(DATA_ROOT, 'image_dhash.pkl')
        print('generate dhash...')
        for path in tqdm(image_path):
            idx = path.split('/')[-1].split('.')[0]
            img = load_image(path)
            dhash = dhash_int(img)
            image_dhash[idx] = dhash
        with open(image_dhash_pickle_file, 'wb') as f:
            pickle.dump(image_dhash, f)
        print('done.')
