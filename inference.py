import torch
from src.open_clip import create_model_and_transforms
import pandas as pd
from src.open_clip.coop_model import COOPCLIP
from PIL import Image
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_parser(): 
    parser = argparse.ArgumentParser(description='EventType')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--class_names_path', default='')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--image_path', default='')
    
    return parser.parse_args()

args_infer = get_parser()

def load_model_coop():
    checkpoint = torch.load(args_infer.checkpoint_path, map_location='cpu')
    clip_model, _, preprocess = create_model_and_transforms(
            model_name= 'ViT-B-32' ,
            pretrained= 'openai' ,
            precision= "amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            pretrained_image=False,
        )
    
    kg_info = None
    class_names = pd.read_csv( args_infer.class_names_path )['gt_label']

    model = COOPCLIP( list( class_names) , kg_info, clip_model,  device) 

    sd = checkpoint["state_dict"]
    if  next(iter(sd.items()))[0].startswith('module'): 
        sd = {k[len('module.'):]: v for k, v in sd.items() if k not in [ 'module.prompt_learner.token_prefix', 'module.prompt_learner.token_suffix'] }

    model.load_state_dict(sd)

    prompts = model.prompt_learner() #n-class, 77, 512
    tokenized_prompts = model.tokenized_prompts  # n-class, 77
    prompt_features = model.text_encoder(prompts, tokenized_prompts)  # n-class, 512 (different per class when CSC=False? give a cls it changes)

    return clip_model, class_names, prompt_features, preprocess

# get event features

print('load model')
clip_model, class_names, prompt_features, preprocess = load_model_coop()


print('load image')
image = preprocess(Image.open(args_infer.image_path)).unsqueeze(0)


print('get similarity')


with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

    text_probs =  image_features @ prompt_features.T
    topk_inds = torch.topk(text_probs, args_infer.topk).indices.squeeze(0).tolist()
    topk_ps = torch.topk(text_probs, args_infer.topk).values.squeeze(0).tolist()

    for i in range(args_infer.topk):
        event_row_id = topk_inds[i]
        event_label = class_names[event_row_id]
        p_event = topk_ps[i]
        event_features = prompt_features[event_row_id]
        print('top', i+1 ,':' , event_label)


