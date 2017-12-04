# coding=utf-8

import os
import pickle
import platform
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing
from utils.configures import *
from utils.read_image import read_image
from model.perceptual_hash.difference_hash import get_num_bits_different, dhash_int


"""
检索图片
"""


# TODO: delete codes below
def query(query_dict, image_dict, method='dhash', top=10):
    """
    搜索相似图片
    :param query_dict: dict, {query_id: encode}
    :param image_dict: dict, {image_id: encode}
    :param method: string, 'dhash', 'vgg19'
    :param top:
    :return:
    """
    # 暴力法
    result = {}

    def compute_similarity(query_id, encoding1, image_id, encoding2, method):
        if method == 'dhash':
            dist = get_num_bits_different(query_encoding, image_encoding)
            if dist <= 50:
                result[query_id].append(image_id)
        elif method == 'vgg19':
            sim = np.dot(query_encoding, image_encoding)
            if sim >= 0.6:
                result[query_id].append(image_id)



    for query_id, query_encoding in tqdm(query_dict.items()):
        if not query_id in result:
            result[query_id] = []
        tmp = []
        p = multiprocessing.Pool(10)
        for image_id, image_encoding in image_dict.items():
            # if method == 'dhash':
            #     sim = 128 - get_num_bits_different(query_encoding, image_encoding)
            # elif method == 'vgg19':
            #     sim = np.dot(query_encoding, image_encoding)
            #
            # if len(tmp) < top:
            #     tmp.append(sim)
            #     result[query_id].append((image_id, sim))
            # else:
            #     if sim <= min(tmp):
            #         continue
            #     else:
            #         idx = np.argmin(tmp)
            #         tmp.remove(tmp[idx])
            #         result[query_id].remove(result[query_id][idx])
            p.apply_async(compute_similarity, args=(query_id, query_encoding, image_id, image_encoding, method, ))
        p.close()
        p.join()

    with open(os.path.join(DATA_ROOT, 'query_result_{}.pkl'.format(method)), 'wb') as f:
        pickle.dump(result, f)
    print('done')
    return result


if 'Linux' in platform.platform():
    num_of_threads = 20
else:
    num_of_threads = 2


def generate_hash_codes(dir_path, pickle_file, hash_method='dhash'):
    """
    generate hash codes for images
    :param dir_path: directory path of images
    :param pickle_file: pickle file name to load or write
    :param hash_method: hash method used, including dhash, vgg19, etc.
    :return:
    """
    if hash_method == 'dhash':
        dhash_dict = {}
        print('generate dhash...')
        p = multiprocessing.Pool(num_of_threads)
        for idx, image in tqdm(read_image(dir_path)):
            dhash_dict[idx] = p.apply_async(dhash_int, args=(image, 8,)).get()
        p.close()
        p.join()
        with open(pickle_file, 'wb') as f:
            pickle.dump(dhash_dict, f)
        print('done.')

        return dhash_dict

    return


def brute_force_search(query_image, image_dict, threshold):
    result = []
    for idx, img in image_dict.items():
        dist = get_num_bits_different(query_image, img)
        if dist <= threshold:
            result.append((idx, dist))
    return result


def image_retrieval(query_dir, threshold, top, hash_method='dhash'):
    query_result = {}
    if hash_method == 'dhash':
        if os.path.exists(QUERY_DHASH):
            with open(QUERY_DHASH, 'rb') as f:
                query_dhash = pickle.load(f)
        else:
            query_dhash = generate_hash_codes(query_dir, QUERY_DHASH, hash_method='dhash')

        with open(IMAGE_DHASH, 'rb') as f:
            image_dhash = pickle.load(f)

        for idx, image in tqdm(query_dhash.items()):
            query_result[idx] = brute_force_search(image, image_dhash, threshold)

        with open(QUERY_RESULT_DHASH, 'wb') as f:
            pickle.dump(query_result, f)

    return query_result


if __name__ == '__main__':

    # TODO: delete codes below
    # with open(os.path.join(DATA_ROOT, 'query_dhash.pkl'), 'rb') as f:
    #     query_dhash = pickle.load(f)
    #
    # with open(os.path.join(DATA_ROOT, 'image_dhash.pkl'), 'rb') as f:
    #     image_dhash = pickle.load(f)
    #
    # qurey_result = query(query_dict=query_dhash, image_dict=image_dhash, method='dhash', top=100)

    query_dir = os.path.join(DATA_ROOT, 'test')
    # query_dhash = generate_hash_codes(dir_path=query_dir, pickle_file=QUERY_DHASH, hash_method='dhash')
    query_result = image_retrieval(query_dir=query_dir, threshold=30, top=10, hash_method='dhash')
    print(query_result)
