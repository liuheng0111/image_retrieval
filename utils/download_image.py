# coding=utf-8


import pandas as pd
import numpy as np
import urllib
import os
import multiprocessing

ROOT = '/data/wangyuan/ImageTextMatching'
koubei_photo_path = os.path.join(ROOT, 'data/koubeiphoto.csv')
img_size = '400x300_0_q87_'
to_path = os.path.join(ROOT, 'data/images')

'''
img_size 取值：
64x48_0_q87_
80x60_0_q87_
120x90_0_q87_
130x98_0_q87_
160x120_0_q87_
200x150_0_q87_
240x180_0_q87_
400x300_0_q87_
800x600_1_q87_
“” # 原图
'''
data_path = ''
for p in os.listdir(os.path.join(ROOT, 'data')):
    if 'koubei-' in p:
        data_path = os.path.join(ROOT, 'data/' + p)
print('data path: '.format(data_path))

if data_path:
    image_df = pd.read_csv(data_path)
else:
    image_df = pd.read_table(koubei_photo_path, header=None)
    image_df.columns = ['img_id', 'kb_id', 'img_key', 'description', 'width', 'height']

    image_df.fillna('', inplace=True)

    image_df = image_df[image_df.img_key.map(lambda x: 'autohomecar__' in x)]
    print('number of images from dfs: {}'.format(image_df.shape[0]))

    # 随机抽取大约 20w 条口碑信息
    _filter = [np.random.random() < 0.4 for _ in range(image_df.kb_id.unique().size)]
    kb_ids = image_df.kb_id.unique()[_filter]
    print('number of koubei randomly selected: {}'.format(kb_ids.size))

    # 抽取对应的图片 id
    kb_ids = pd.DataFrame(kb_ids)
    kb_ids.columns = ['kb_id']
    image_df = kb_ids.merge(image_df, on='kb_id', how='left')
    print('number of images: {}'.format(image_df.shape[0]))

    # 保存数据
    image_df.to_csv(os.path.join(ROOT, 'data/koubei-{}.csv'.format(kb_ids.size)), index=False)

image_df = image_df[17523:]
print('{} images to download.'.format(image_df.shape[0]))


def saveImage(img_key, img_id, img_size, to_path_root):
    for k in range(4)[::-1]:
        http_base = 'http://k{}.autoimg.cn/'.format(k)
        img_url = http_base + img_key.replace('autohomecar__', img_size + 'autohomecar__')
        to_path = os.path.join(to_path_root, str(img_id) + '.jpg')
        try:
            # print(img_url)
            # print(to_path)
            urllib.request.urlretrieve(img_url, to_path)
            print('good.')
            break
        except:
            print('invalid image url {}'.format(img_url))
            continue


def getImages(img_df, img_size, to_path_root):
    cnt = 17512
    p = multiprocessing.Pool(35)
    for i in img_df.index:
        img_key = img_df.loc[i]['img_key']
        img_id = img_df.loc[i]['img_id']
        cnt += 1
        if cnt % 100 == 0:
            print('{} images downloaded.'.format(cnt))
        p.apply_async(saveImage, args=(img_key, img_id, img_size, to_path_root))
        # print(i)
    p.close()
    p.join()
    print('done.')


if __name__ == '__main__':
    getImages(image_df, img_size, to_path)
