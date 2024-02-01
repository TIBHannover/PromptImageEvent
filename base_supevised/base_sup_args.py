import argparse
import os
root = os.path.dirname(os.path.abspath(__file__))


def get_parser_sup(): 
    parser = argparse.ArgumentParser(description='EventType')

    parser.add_argument('--dataset', default='vise_bing', help='vise_bing,vise_wiki,instances,red,wider,soceid' )
    parser.add_argument('--checkpoints_path', default=f'{root}/checkpoints' )
    parser.add_argument('--data_path_', default=f'{root}/../datasets' )
    parser.add_argument('--caption_chk_path',  default=f'{root}/checkpoints/caption_chk.pt' )
    parser.add_argument('--clip_pretrained', default='openai' )
    parser.add_argument('--clip_model_name', default='ViT-B-32' )
    parser.add_argument('--model_type', default='linear', help='reg,linear' )
    parser.add_argument('--path_results', default=f'{root}/supervised/results' )
    parser.add_argument('--model_pretrained', default='caption', help='vanilla,caption' )
    

    args = parser.parse_args()

    return args

