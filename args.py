import argparse
import os
root = os.path.dirname(os.path.abspath(__file__))


def get_parser(): 
    parser = argparse.ArgumentParser(description='EventType')
    parser.add_argument('--dataset', default='soceid', help='vise_bing,vise_wiki,instances,red,wider,soceid')
    parser.add_argument('--test_images_path', default='')
    parser.add_argument('--query_text_aggregation', default='max', help='max,mean')
    parser.add_argument('--graph_name', default='refined', help='disambiguated')
    parser.add_argument('--data_path', default=f'{root}/data')
    parser.add_argument('--results_path', default=f'{root}/results')
    parser.add_argument('--clip_model_name', default= 'ViT-B-32', help='ViT-B-32 , RN50, RN101')
    parser.add_argument('--kg_info_path', default='')
    parser.add_argument('--clip_pretrained', default= 'openai', help = ' ViT: [ openai, laion400m_e32 ] , RN: [ yfcc15m , openai ] ' )
    parser.add_argument('--graph_path', default=f'{root}/data/vise/VisE-O_refined.graphml')
    parser.add_argument('--class_names_path', default=f'{root}/data/vise/class_names.csv')
    parser.add_argument('--inf_strategy', default = 'leaf', help = 'leaf/branch')  
    parser.add_argument( "--kg_init",  default='random',   help="wikidata,wikipedia,random")
    parser.add_argument( "--CSC",  default='True',  help="")
    parser.add_argument( "--CLASS_TOKEN_POSITION", default= 'front',  help="front,middle,end")
    parser.add_argument( "--N-CTX", default= 16 )
    parser.add_argument( "--ctx_init", default="" ,help="This is a photo of a" )
    parser.add_argument( "--pre_fname", default=f"" )
    parser.add_argument( "--pre_path_checkpoints", default =f'{root}/checkpoints')
    parser.add_argument( "--shot", default= 30 )
    parser.add_argument( "--model_name", default= 'coop' ,help='vanilla,caption,coop,coop_caption' )
    parser.add_argument( "--caption_checkpoint", default= f'{root}/checkpoints/caption_pretrained_chk.pt' )

    args = parser.parse_args()

    return args

