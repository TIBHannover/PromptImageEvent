import torch
from PIL import Image
import time
import h5py
import os
import json
from args import get_parser
from pathlib import Path
import pandas as pd
args = get_parser()


def embed_all_images(images_data, model, preprocess, images_path, image_emb_output_dir):

    def get_single_image_emb(path_image, preprocess, model):
        if not os.path.exists(path_image):
            path_image = f'{args.data_path}/{args.dataset}/images/{path_image}'

        image = preprocess(Image.open(path_image)).unsqueeze(0)
        
        with torch.no_grad():  image_emb = model.encode_image(image)

        return image_emb.squeeze().detach().cpu().numpy()

    tic = time.perf_counter()
    Path( os.path.dirname(image_emb_output_dir)).mkdir(parents=True, exist_ok = True)
    h5f = h5py.File(image_emb_output_dir, 'w')

    for e_counter, ev_id in enumerate(images_data):
        images = images_data[ev_id]

        for im_counter, img in enumerate(images):

            id = img['image_hash']
            img_label = img['leaf_wd_label']
            grp = h5f.create_group(str(id))
            img_path = f'{images_path}/{img["image_path"]}'
            image_emb = get_single_image_emb(img_path,  preprocess, model)

            grp.create_dataset(name = 'clip', data = image_emb, compression="gzip", compression_opts=9)
            grp.create_dataset(name = 'gt_id', data = [ev_id], compression="gzip", compression_opts=9)
            grp.create_dataset(name = 'gt_label', data = [img_label], compression="gzip", compression_opts=9)
            grp.create_dataset(name = 'image_path', data = [img_path], compression="gzip", compression_opts=9)            
            
            toc = time.perf_counter()
            print(f'Getting image embeddings ====> event: {e_counter + 1} |  {im_counter + 1}/{ len(images_data[ev_id]) } images -- {toc - tic} s ')

    h5f.close()
    print('done')

def get_image_paths():  
    data = {}
    print('... Getting images for ', args.dataset)

    if args.dataset in ['vise_bing', 'vise_wiki']:

        if args.dataset == 'vise_bing': jsonl_path = f'{args.data_path}/vise/test_VisE-Bing.jsonl'
        else: jsonl_path = f'{args.data_path}/vise/test_VisE-Wiki.jsonl'
        with open(jsonl_path) as f:
            for line in f:
                line = json.loads(line)
                cls = line['leaf_wd_id']

                if cls in data: data[cls].append(line)
                else:  data[cls] = [line]
    
    elif args.dataset == 'instances':
        data_file = pd.read_csv(args.test_data_instance , sep=',')

        for i in range(len(data_file)):
            row =  data_file.iloc[i] 
            cls = row['gt_id']
            line = {'gt_id':row['gt_id'], 'leaf_wd_label': row['gt_label'],   'image_path': row['filepath'],    'image_hash': row['filepath'].split('/')[-1] }
            if cls in data: data[cls].append( line )
            else:  data[cls] = [ line ]

    else:

        if args.dataset == 'red':  jsonl_path = f'{args.data_path}/{args.dataset}/test_RED.jsonl'
        elif args.dataset == 'wider':  jsonl_path = f'{args.data_path}/{args.dataset}/test_WIDER.jsonl'
        elif args.dataset == 'soceid':  jsonl_path = f'{args.data_path}/{args.dataset}/test_SocEID.jsonl'

        with open(jsonl_path) as f:
            for line in f:
                line = json.loads(line)
                cls = line['leaf_wd_id']

                if cls in data: data[cls].append(line)
                else:  data[cls] = [line]
    return data


def get_image_emb(images_path, model, preprocess, events, h5_image_path):
    
    event_ids = list(events.keys())
    
    if not os.path.exists(h5_image_path):  
        print('saving image embs ...')
        images_data = get_image_paths()
        image_h5 = embed_all_images(images_data, model, preprocess, images_path, h5_image_path)

    # get matrix of ground truth: M_gt, n-class one-hot encode
    M_i = []
    M_gt = []
    image_h5 = h5py.File(h5_image_path, 'r')
    
    for id_img in image_h5:
        gt = image_h5[id_img]['gt_id'][()][0].decode('utf-8')
        M_gt.append(event_ids.index(gt))
        M_i.append( image_h5[id_img]['clip'][()].squeeze() )

    print('******* # test images: ', len(M_gt))
    return M_i, M_gt

