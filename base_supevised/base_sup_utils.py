import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, root)
from src.open_clip import create_model_and_transforms
import json
import torch
import time
from pathlib import Path
import pandas as pd
import h5py
from PIL import Image
from base_sup_args import get_parser_sup

args = get_parser_sup()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_data_path(n_shot, n_round, dataset, data_path_, model_pretrained):
    if dataset == 'vise_bing': 
        train_path = f'{data_path_}/vise/train/round{n_round}/vise_train_{n_shot}_shot_round_{n_round}.csv'
        val_path = f'{data_path_}/vise/val/round{n_round}/vise_val_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/vise/test_VisE-Bing.jsonl'  
        class_names_path =  f'{data_path_}/vise/class_names.csv' 
        graph_path = f'{data_path_}/vise/VisE-O_refined.graphml'
        test_path_image_h5 = f'{data_path_}/{dataset}/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/vise/train/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/vise/val/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    elif dataset == 'vise_wiki': 
        train_path = f'{data_path_}/vise/train/round{n_round}/vise_train_{n_shot}_shot_round_{n_round}.csv'
        val_path =   f'{data_path_}/vise/val/round{n_round}/vise_val_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/vise/test_VisE-Wiki.jsonl'
        class_names_path =  f'{data_path_}/vise/class_names.csv'
        graph_path = f'{data_path_}/vise/VisE-O_refined.graphml'
        test_path_image_h5 = f'{data_path_}/{dataset}/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/vise/train/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/vise/val/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    elif dataset == 'red': 
        train_path = f'{data_path_}/red/train/round{n_round}/train_{n_shot}_shot_round_{n_round}.csv'
        val_path = f'{data_path_}/red/val/round{n_round}/val_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/red/test_RED.jsonl'
        class_names_path = f'{data_path_}/red/class_names.csv' 
        graph_path = f'{data_path_}/red/ontology_RED.graphml'
        test_path_image_h5 = f'{data_path_}/red/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/red/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/red/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    elif dataset == 'wider': 
        train_path = f'{data_path_}/wider/train/round{n_round}/train_{n_shot}_shot_round_{n_round}.csv'
        val_path = f'{data_path_}/wider/val/round{n_round}/val_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/wider/test_WIDER.jsonl'
        class_names_path = f'{data_path_}/wider/class_names.csv' 
        graph_path = f'{data_path_}/wider/ontology_Wider.graphml'
        test_path_image_h5 = f'{data_path_}/wider/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/wider/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/wider/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    elif dataset == 'soceid': 
        train_path = f'{data_path_}/soceid/train/round{n_round}/train_{n_shot}_shot_round_{n_round}.csv'
        val_path = f'{data_path_}/soceid/val/round{n_round}/val_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/soceid/test_SocEID.jsonl'
        class_names_path = f'{data_path_}/soceid/class_names.csv' 
        graph_path = f'{data_path_}/soceid/ontology_SocEID.graphml'
        test_path_image_h5 = f'{data_path_}/soceid/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/soceid/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/soceid/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    elif dataset == 'instances': 
        train_path = f'{data_path_}/instances/train/round{n_round}/train_{n_shot}_shot_round_{n_round}.csv'
        val_path = f'{data_path_}/instances/train/round{n_round}/train_{n_shot}_shot_round_{n_round}.csv'
        test_path = f'{data_path_}/instances/test.csv'
        class_names_path = f'{data_path_}/instances/class_names.csv' 
        graph_path = ''
        test_path_image_h5 = f'{data_path_}/instances/ViT-B-32/openai/test_{model_pretrained}_image.h5'
        train_path_h5 = f'{data_path_}/instances/ViT-B-32/openai/train_{model_pretrained}_image_{n_round}.h5'
        val_path_h5 = f'{data_path_}/instances/ViT-B-32/openai/val_{model_pretrained}_image_{n_round}.h5'
    return {'raw': [train_path, val_path, test_path, class_names_path, graph_path], 'h5':[train_path_h5, val_path_h5, test_path_image_h5 ]}




def open_json(fileName):
    with open(fileName,encoding='utf8') as json_data:
        return json.load(json_data)


def load_model_vanilla():
    
    clip_model, _, preprocess = create_model_and_transforms(
            model_name= args.clip_model_name ,
            pretrained= args.clip_pretrained , #openai
            precision= "amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
        )
    
    return clip_model,  preprocess

def load_model_caption(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model, _, preprocess = create_model_and_transforms(
            model_name=args.clip_model_name ,
            pretrained=args.clip_pretrained , 
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
        )
    
    sd = checkpoint["state_dict"]
    if  next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    return model, preprocess


def embed_all_images( csv_path, image_emb_output_dir, caption_chk_path ):
    if args.model_pretrained == 'vanilla':
        model, preprocess = load_model_vanilla()
    elif args.model_pretrained == 'caption':
        model, preprocess = load_model_caption(caption_chk_path)

    def get_single_image_emb(path_image, preprocess, model):
        image = preprocess(Image.open(path_image)).unsqueeze(0)
        with torch.no_grad():
            image_emb = model.encode_image(image)

        return image_emb.squeeze().detach().cpu().numpy()

    tic = time.perf_counter()
    Path( os.path.dirname(image_emb_output_dir)).mkdir(parents=True, exist_ok = True)
    h5f = h5py.File(image_emb_output_dir, 'w')

    if 'csv' in csv_path:  csv_file = pd.read_csv(csv_path)
    elif 'jsonl' in csv_path:  csv_file = pd.read_json(csv_path, lines=True)
    print('csv_path',csv_path)
    
    if 'vise' in args.dataset:
        ids = [ f.split('/')[-1].split('.')[0] for f in csv_file['filepath'] ]
    else:
        ids = [ f.split('/')[-1].split('.')[0]+'+'+f.split('/')[-2] for f in csv_file['filepath'] ]
    

    for irow, id in zip (range( len( csv_file.values) ) , ids):

        row = csv_file.iloc[irow]

        img_label = row['gt_label']
        
        if args.dataset== 'instances':
            img_path = row['filepath']
        else:
            img_path = f"{args.data_path_}/{args.dataset}/images/{row['filepath']}"
        image_emb = get_single_image_emb(img_path, preprocess, model)

        grp = h5f.create_group(str(id))
        grp.create_dataset(name = 'clip', data = image_emb, compression="gzip", compression_opts=9)
        # grp.create_dataset(name = 'gt_id', data = [ev_id], compression="gzip", compression_opts=9)
        grp.create_dataset(name = 'gt_label', data = [img_label], compression="gzip", compression_opts=9)
        
        toc = time.perf_counter()
        print(f'Getting image embeddings ====>  {irow + 1}/{ len(csv_file) } images -- {toc - tic} s ')

    h5f.close() 
    print('done')

def get_image_embeddings( image_h5_path, caption_chk_path, data_split, n_round='', csv_path='', valid_ids = ''):
    print(image_h5_path)
    if not os.path.exists( image_h5_path ): 
        print('embeddings do not exist ...') 
        paths_ = get_data_path(n_shot=30, n_round=n_round, dataset=args.dataset, data_path_=args.data_path_, model_pretrained = args.model_pretrained)  #50 
        [train_path_h5, val_path_h5, test_path_h5] = paths_['h5']
        [train_path, val_path, test_path, _, _] = paths_['raw']
        if data_split == 'train': 
            image_h5_path = train_path_h5
            csv_path = train_path
        elif data_split == 'val': 
            image_h5_path = val_path_h5
            csv_path = val_path
        else:
            csv_path = test_path
            image_h5_path = test_path_h5
        embed_all_images( csv_path, image_h5_path, caption_chk_path )

    image_h5 = h5py.File( image_h5_path )
   
    M_i = []
    M_gt = []
    img_ids = []
  
    if valid_ids !='': valid_ids_in = valid_ids
    else:  valid_ids_in = image_h5


    for n_, id_img in enumerate(valid_ids_in):
      
        M_i.append( image_h5[id_img]['clip'][()].squeeze() )
        gt_label = image_h5[id_img]['gt_label'][()][0].decode('utf-8')

        M_gt.append(gt_label)
        img_ids.append(id_img)

    return M_i, M_gt, img_ids

