import numpy as np
from sklearn.metrics import top_k_accuracy_score, confusion_matrix
import sklearn
import csv
from pathlib import Path
import pandas as pd
import os
import glob
from utils import *
from args import get_parser
from embed_image import get_image_emb, get_image_paths
import torch

args = get_parser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.open_clip import create_model_and_transforms
from embed_prompt import get_prompt_emb
from src.open_clip.coop_model import COOPCLIP



def get_data_paths():

    if 'vise' in args.dataset:  
        args.kg_info_path = f'{args.data_path}/vise/kg_info.json'
        args.class_names_path = f'{args.data_path}/vise/class_names.csv'
        args.test_images_path  = f'{args.data_path}/{args.dataset}/images'

    elif args.dataset == 'red': 
        args.class_names_path = f'{args.data_path}/red/class_names.csv'
        args.kg_info_path = f'{args.data_path}/red/kg_info.json'
        args.test_images_path  = f'{args.data_path}/{args.dataset}/images'

    elif args.dataset == 'wider': 
        args.class_names_path = f'{args.data_path}/wider/class_names.csv'
        args.kg_info_path = f'{args.data_path}/wider/kg_info.json'
        args.test_images_path  = f'{args.data_path}/{args.dataset}/images'
        
    elif args.dataset == 'soceid': 
        args.class_names_path = f'{args.data_path}/soceid/class_names.csv'
        args.kg_info_path = f'{args.data_path}/soceid/kg_info.json'
        args.test_images_path  = f'{args.data_path}/{args.dataset}/images'

    else: # instances
        args.class_names_path = f'{args.data_path}/instances/class_names.csv'
        args.kg_info_path = f'{args.data_path}/instances/kg_info.json'
        args.test_images_path  = f'{args.data_path}/instances/images'
    
    return args

args = get_data_paths()


def get_acc( M_cos_sim, M_gt, query_key= ''):
    
    acck = {}
    for k in [1, 3 , 5]: 
        acck[k] = np.round (top_k_accuracy_score(M_gt, M_cos_sim.T, k=k) * 100, 2 )  # n-samples,  n x 148
    print(query_key, acck)
    
    return acck

def get_acc_per_event(M_cos_sim, M_gt):

    M_pred = np.argmax(M_cos_sim.T, -1) # n-samples,

    matrix = confusion_matrix(M_gt, M_pred)

    acc_events_ = matrix.diagonal()/matrix.sum(axis=1) * 100 # n-samples, n-class

    return np.round( acc_events_, 2 )  

def get_queries():
    qLS_queries = []
    for iq in range(80):  qLS_queries.append('extended-' + str(iq))

    queries_file = {'Q-LL': ['class-name'], 'Q-LS': ['openai'] , 'Q-WD':['wikidata'] , 'Q-WP':['wikipedia'],     
                    'Q-WD-WP':['wikidata', 'wikipedia'] , 'Q-LL-LS':['class-name','openai'], 'Q-LL-WD':['class-name', 'wikidata'] , 'Q-LL-WP':['class-name', 'wikipedia'] , 'Q-LS-WD': ['openai','wikidata'], 'Q-LS-WP': ['wikipedia','openai'] ,          
                    'Q-LL-LS-WD': ['class-name', 'openai','wikidata'], 'Q-LL-LS-WP':['class-name','openai','wikipedia'],  'Q-LS-WD-WP': ['openai','wikidata','wikipedia'] , 'Q-LL-LS-WD-WP': ['class-name', 'openai','wikidata', 'wikipedia'] }
    # queries_file = { 'Q-LL': ['class-name'], 'Q-LS': ['openai'] , 'Q-LL-LS':['class-name','openai'],
    #                  'Q-WD':['wikidata'] , 'Q-WP':['wikipedia'],
    #                  'Q-WD-WP':['wikidata', 'wikipedia'], 'Q-LS-WD-WP': ['openai','wikidata','wikipedia'], 
    #                  'Q-LL-LS-WD-WP': ['class-name', 'openai','wikidata', 'wikipedia']
    #                 }
    
    return queries_file

def operate(ls, op, axis=0):

    dic_op = {'max': np.max(ls,axis), 'mean': np.mean(ls,axis)}

    return dic_op[op]

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
    
    return clip_model, [], [], preprocess

def load_model_caption():
    checkpoint = torch.load(args.caption_checkpoint, map_location='cpu')
    clip_model, _, preprocess = create_model_and_transforms(
            model_name= args.clip_model_name ,
            pretrained= args.clip_pretrained , #openai
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
        )
    
    sd = checkpoint["state_dict"]
    if  next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    clip_model.load_state_dict(sd)

    return clip_model, [], [], preprocess

def load_model_coop(checkpoint_path, shot):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    clip_model, _, preprocess = create_model_and_transforms(
            model_name= args.clip_model_name ,
            pretrained= args.clip_pretrained , #openai
            precision= "amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
        )
    
    if os.path.exists(args.kg_info_path):
        with open(args.kg_info_path, encoding='utf8') as f_kg:
            kg_info = json.load(f_kg)
    else: kg_info = None

    class_names = pd.read_csv(args.class_names_path)['gt_label']

    model = COOPCLIP( list( class_names) , kg_info, clip_model,  device) 

    sd = checkpoint["state_dict"]
    if  next(iter(sd.items()))[0].startswith('module'): 
        sd = {k[len('module.'):]: v for k, v in sd.items() if k not in [ 'module.prompt_learner.token_prefix', 'module.prompt_learner.token_suffix'] }
    # print(model.prompt_learner)
    if args.CSC == 'False':
        sd['prompt_learner.token_prefix'] = model.prompt_learner.token_prefix #model.prompt_learnevise_wikiompt_learner.token_suffix
        sd['prompt_learner.token_suffix'] = model.prompt_learner.token_suffix #model.prompt_learnevise_wikiompt_learner.token_suffix
    model.load_state_dict(sd)
    ##
    prompts = model.prompt_learner() #n-class, 77, 512
    tokenized_prompts = model.tokenized_prompts  # n-class, 77
    prompt_features = model.text_encoder(prompts, tokenized_prompts)  # n-class, 512 (different per class when CSC=False? give a cls it changes)
    ##

    if args.model_name == 'coop_caption':
        clip_model, _, _, _ = load_model_caption()

    return clip_model, class_names, prompt_features, preprocess


def get_sims(query_key, queries_file, M_im, emb_txt, event_ids):
    queries = queries_file[query_key]
    prompt_features = []

    for q in queries:
        if q in ['openai']:  
            prompt_features.append( operate( emb_txt[q] , args.query_text_aggregation , axis=1) )
        else:   
            prompt_features.append( emb_txt[q] )

    M_im = np.stack(M_im, axis=0)  # n, d
    prompt_features = np.stack(prompt_features, axis=0)  # n_query, n-class, 512 

    M_sims = []

    for eind in range( len(event_ids)):
        prompt_feature = prompt_features[:, eind, :]  # n_query, 512
        
        M_cos_sim = 1 - sklearn.metrics.pairwise.cosine_distances(M_im, prompt_feature )  # n, 512 .  n_query, 512  =  n, n_query

        if M_cos_sim.shape[-1] == 1: M_cos_sim = M_cos_sim.squeeze(-1)
        else: M_cos_sim = operate(M_cos_sim, args.query_text_aggregation, axis=1)  # n,
        M_sims.append(M_cos_sim)

    M_sims = np.stack(M_sims, 0) # n-class, n
    return M_sims

def save_acc_total(path_results, acc_total):
 
    print('saving results')
    queries_file = get_queries()
    f = open(path_results + '/acc_total.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Topk']+ list(queries_file.keys()) )
    for k in [1,3,5]:
        row = [k]
        for q in acc_total:
            row+= [acc_total[q][k]]
        writer.writerow(row)
    f.close()

def save_acc_events(path_results, all_events, acc_events):

    df = pd.DataFrame(acc_events)
    df['Events'] = list(all_events.values())
    df.to_csv(path_results + '/acc_events.csv')

def get_results(emb_txt_leaf, emb_txt_branch, leaf_events, branch_events, M_im, M_gt, Mapper, path_results):
    # get matrix of features for images: M_i: N x D
    # get matrix of features for events: M_e: n-class x D

    queries_file = get_queries()
    
    # get acc values
    acc_total = {}
    acc_events = {}  

    for query_key in queries_file:

        M_cos_sim_leaf = get_sims(query_key, queries_file, M_im, emb_txt_leaf, list(leaf_events.keys()))  # leaf, n

        if args.inf_strategy == 'branch': 
            M_cos_sim_branch0 = get_sims(    query_key, queries_file, M_im, emb_txt_branch, list(branch_events.keys())   ) # branch, n
            M_cos_sim_branch = [] # n, branch

            for i_im in range(len(M_im)):
                l_size = np.shape( Mapper)[0]
                
                M_cos_sim_temp = np.repeat(M_cos_sim_branch0[:, i_im].expand, l_size, axis = 1) # 1, leaf
                M_cos_sim_branch0[i_im] = M_cos_sim_temp * Mapper.T    # b, leaf
                if args.branch_op == 'mean':    s = np.mean(M_cos_sim_branch0[i_im], axis=1)  # branch,
                elif args.branch_op == 'max':   s = np.max(M_cos_sim_branch0[i_im], axis=1)  # branch,
                M_cos_sim_branch.append(s)
            M_cos_sim_branch = np.stack(M_cos_sim_branch, 0)

            if args.branch_to_leaf_op == 'mean':     M_cos_sim = np.mean( np.array([ M_cos_sim_branch, M_cos_sim_leaf ]), axis=0 )
            elif args.branch_to_leaf_op == 'max':    M_cos_sim = np.max(  np.array([ M_cos_sim_branch, M_cos_sim_leaf ]), axis=0 )
        else:
            M_cos_sim = M_cos_sim_leaf

        acc_total[query_key] = get_acc( M_cos_sim, M_gt ,query_key )  
        acc_events[query_key] = get_acc_per_event(M_cos_sim, M_gt)  

    return acc_total, acc_events


leaf_events0 = get_events(node_type = 'leaf', args=args)
## re-arrange leaf events based on class names in the model
class_names = pd.read_csv(args.class_names_path)['gt_label']
leaf_events = {}
for clsname in class_names:
    for e_id in leaf_events0:
        if leaf_events0[e_id] == clsname:
            leaf_e_id = e_id
            break
    leaf_events[leaf_e_id] = clsname
##
[branch_events] = [get_events(node_type = 'branch') if args.inf_strategy == 'branch' else None]

# load model func
if args.model_name in ['coop', 'coop_caption']:   load_model = load_model_coop
elif args.model_name == 'caption': load_model = load_model_caption
else:   load_model = load_model_vanilla

if os.path.exists(args.kg_info_path):  kg_info = open_json(args.kg_info_path)  # wikidata, wikipedia per event id
else: kg_info = []

# TEST DATA: get image embs: doesnt matter which shot
if args.model_name in ['caption', 'coop_caption']:    test_image_h5_file =  f'{args.data_path}/{args.dataset}/{args.clip_model_name}/{args.clip_pretrained}/test_caption_image.h5'
else: test_image_h5_file =  f'{args.data_path}/{args.dataset}/{args.clip_model_name}/{args.clip_pretrained}/test_vanilla_image.h5'
clip_model = []
preprocess = []

args.pre_fname = f'{args.pre_fname}_vise' if 'vise' in args.dataset else f'{args.pre_fname}_{args.dataset}'
rounds = [1,2,3]
shots = [args.shot] if args.shot !=-1 else [30]
META_acc_total = {}
META_acc_events = {}


if args.model_name in ['vanilla', 'caption']: 

    # make results directory
    path_results = f'{args.results_path}/{args.model_name}/{args.dataset}/{args.inf_strategy}/{args.query_text_aggregation}'
    Path(path_results).mkdir(parents=True, exist_ok= True)
    print(path_results)

    # load model
    clip_model, class_names, prompt_features, preprocess = load_model()

    # event prompts
    prompt_emb_leaf = get_prompt_emb(leaf_events, prompt_features, class_names, clip_model, args.model_name, kg_info=kg_info)
    
    [prompt_emb_branch] = [get_prompt_emb(branch_events, prompt_features, class_names, clip_model, args.model_name, kg_info=kg_info) if args.inf_strategy == 'branch' else None]

    M_im, M_gt = get_image_emb(args.test_images_path, clip_model, preprocess, leaf_events, test_image_h5_file)  # classes

    # save results
    acc_total, acc_events = get_results(prompt_emb_leaf, prompt_emb_branch, leaf_events, branch_events, M_im, M_gt, [], path_results)
    
    for q in acc_total:
        if q not in META_acc_total:
            META_acc_total[q] = {1:[], 3:[], 5:[]}
            META_acc_events[q] = []
        for _k in [1,3,5]:  META_acc_total[q][_k].append( acc_total[q][_k] ) # n-rounds, n-samples
        META_acc_events[q].append( acc_events[q] ) # n-rounds, n-samples, n-class

else:

    for shot_ in shots:

        for round_ in rounds:
            print('** round', round_)
        
            print(f'loading model {args.model_name} shot', shot_)
            chk_folder_name = f'{args.pre_fname}-init-{args.kg_init}-model-{args.clip_model_name}-ctx-{args.N_CTX}-{args.CLASS_TOKEN_POSITION}-shot-{shot_}-round-{round_}'
            chk_path = f'{args.pre_path_checkpoints}/{chk_folder_name}' 
            path_results = f'{args.results_path}/{args.model_name}_{args.pre_fname}-init-{args.kg_init}-model-{args.clip_model_name}-ctx-{args.N_CTX}-{args.CLASS_TOKEN_POSITION}-shot-{shot_}/{args.dataset}/{args.inf_strategy}/{args.query_text_aggregation}'
            print(chk_path)
            ## make results directory
            Path(path_results).mkdir(parents=True, exist_ok= True)
            text_checkpoint_path = glob.glob(f'{chk_path}/checkpoints/*.pt')[0]   
            print(text_checkpoint_path)
            
            ## load model
            clip_model, class_names, prompt_features, preprocess = load_model(text_checkpoint_path, shot_)
            
            ## Test image embeddings
            M_im, M_gt = get_image_emb(args.test_images_path, clip_model, preprocess, leaf_events, test_image_h5_file)  

            ## event prompts
            prompt_emb_leaf = get_prompt_emb(leaf_events, prompt_features, class_names, clip_model, args.model_name, kg_info=kg_info)
            prompt_emb_branch = [ get_prompt_emb(branch_events, prompt_features, class_names, clip_model, args.model_name) if args.inf_strategy == 'branch' else None]
            
            ## save results
            acc_total, acc_events = get_results(prompt_emb_leaf, prompt_emb_branch, leaf_events, branch_events, M_im, M_gt, [], path_results)
            
            for q in acc_total:
                if q not in META_acc_total:
                    META_acc_total[q] = {1:[], 3:[], 5:[]}
                    META_acc_events[q] = []
                for _k in [1,3,5]:  META_acc_total[q][_k].append( acc_total[q][_k] ) # n-rounds, n-samples
                META_acc_events[q].append( acc_events[q] ) # n-rounds, n-samples, n-class

# save per shot - mean all rounds
for q in META_acc_total:
    META_acc_events[q] = np.round( np.mean( np.array(META_acc_events[q]) ,axis = 0),2 )
    
    for _k in [1,3,5]:
        META_acc_total[q][_k] = np.round(  np.mean(np.array( META_acc_total[q][_k]) ,axis = 0), 2 )

save_acc_total(path_results, META_acc_total)
save_acc_events(path_results, leaf_events, acc_events)

