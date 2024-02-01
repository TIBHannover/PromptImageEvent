import time
import torch
from src.open_clip import tokenize
from utils import openai_imagenet_template, get_all_event_labels
import numpy as np
from args import get_parser
args = get_parser()


def get_event_prompts( all_events, kg_info):
    event_ids = list(all_events.keys())

    output = {'class-name':{}, 
              'openai':{},
              'wpedia': {}, 'wdata': {}
              }

    for ev_id0 in event_ids:
        ev_label0 = all_events[ev_id0]
        output['class-name'][ev_id0] = ev_label0

        output['openai'][ev_id0] = []
        for i_ex, ext_text_func in enumerate( openai_imagenet_template() ): 
            output['openai'][ev_id0].append( ext_text_func(ev_label0) )

        ev_id = ev_id0.split('/')[0]
        ev_label = ev_label0.split('/')[0]

        output['wpedia'][ev_id] = f'This is a photo of a {ev_label}. {kg_info[ev_label0]["wikipedia"]}'
        output['wdata'][ev_id] = f'{ev_label} is a {kg_info[ev_label0]["wikidata"]}' 

    return output

def get_prompt_emb(all_events, prompt_features, class_names, clip_model, model_name, kg_info): # only class-name

    def get_text_emb_clip_0( text_to_embed,  clip_model):

        tkns = tokenize(text_to_embed)
        with torch.no_grad():
            sent_emb0 = clip_model.encode_text(tkns ).squeeze()
        total_embs = sent_emb0.squeeze().detach().cpu().numpy()
    
        return total_embs
    

    def get_text_emb_coop( class_label, class_names, prompt_features):

        event_classname_rows =  class_names.loc[class_names == class_label].index[0]
        emb0 = torch.index_select( prompt_features, 0, torch.Tensor( [event_classname_rows]).to(int) )        
        emb = emb0.squeeze().detach().cpu().numpy()

        return emb

    event_prompts = get_event_prompts(all_events, kg_info)  # class-name, openai-prompts 
    data_output = {}
    data_output['openai'] = []
    data_output['class-name'] = []
    data_output['wikidata'] = []
    data_output['wikipedia'] = []
    t0 = time.time()
    event_ids = list(all_events.keys())


    for ie, ev_id0 in enumerate( event_ids[:]):
        print('Getting event embeddings...', ie+1, len(event_ids), '| time: ', time.time()-t0)

        data_output['openai'].append( get_text_emb_clip_0(text_to_embed = event_prompts['openai'][ev_id0], clip_model = clip_model) )
        
        if 'coop' in model_name:  data_output['class-name'].append( get_text_emb_coop(class_label = event_prompts['class-name'][ev_id0], class_names= class_names, prompt_features = prompt_features))
        else:   data_output['class-name'].append( get_text_emb_clip_0(text_to_embed= event_prompts['class-name'][ev_id0], clip_model=clip_model))
        
        data_output['wikidata'].append( get_text_emb_clip_0(text_to_embed = list(event_prompts['wdata'].values())[ie], clip_model = clip_model)  )
        data_output['wikipedia'].append( get_text_emb_clip_0(text_to_embed = list(event_prompts['wpedia'].values())[ie], clip_model = clip_model)  )
        
    data_output['openai'] = np.stack(data_output['openai'], axis=0)
    data_output['class-name'] = np.stack(data_output['class-name'], axis=0)
    data_output['wikidata'] = np.stack(data_output['wikidata'], axis=0)
    data_output['wikipedia'] = np.stack(data_output['wikipedia'], axis=0)
    
    return data_output



