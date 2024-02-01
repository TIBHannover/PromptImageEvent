import numpy as np
import pandas as pd
import networkx as nx
import csv 
import json
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, root_dir)
from embed_image import get_image_emb, get_image_paths


def get_all_event_labels(node_type, graph_path):
    all_events = {}
    graph_in = nx.read_graphml(graph_path)
    
    for i, node in enumerate( graph_in.nodes(data=True) ):
        if node[1]['node_type'] == node_type:
            node_id = node[1]['wd_id']
            node_label = node[1]['wd_label']
            all_events[node_id] = node_label

    return all_events


def get_events(node_type, args):  # {event_id: event_label}
    if args.dataset in ['vise_bing', 'vise_wiki']:
        graph_path = f'{args.data_path}/vise/VisE-O_{args.graph_name}.graphml'
        all_events = get_all_event_labels(node_type, graph_path)
        
        if args.dataset == 'vise_wiki':
            image_data = get_image_paths()
            all_events_ = {}
            for e in all_events: # has 146 events
                if e in list(image_data.keys()):
                    all_events_[e] = all_events[e]
            return all_events_
        else:
            return all_events
    elif args.dataset == 'instances':
        data_test = pd.read_csv(args.test_data_instance)
        gt_labels = list(data_test['gt_label'])
        gt_ids =list( data_test['gt_id'])
        all_events = {}
        _labels = {}
        for i in range(len(gt_ids)):
            if gt_labels[i] not in _labels:
                all_events[gt_ids[i]] = gt_labels[i]
                _labels[gt_labels[i]] = gt_ids[i]

        return all_events
    else:
        if args.dataset == 'red': graph_name = 'ontology_RED'
        elif args.dataset == 'soceid': graph_name = 'ontology_SocEID'
        elif args.dataset == 'wider': graph_name = 'ontology_WIDER'

        graph_path = f'{args.data_path}/{args.dataset}/{graph_name}.graphml'
        all_events = get_all_event_labels(node_type , graph_path)
        
        return all_events

def get_all_branch_nodes(graph_in):
    G_node_labels = {}
    for i,node in enumerate( graph_in.nodes(data=True) ):
        print(i, 'getting branch nodes ...')
        node_id = node[1]['wd_id']
        node_label = node[1]['wd_label']
        G_node_labels[node_id] = node_label
    return G_node_labels



def save_csv(input_dic, fname):
        
        f = open(fname, 'w')
        writer = csv.writer(f)

        for k in input_dic:
                if isinstance(input_dic[k], list):
                    row = [k] + input_dic[k]
                else:
                    row = [k] + [input_dic[k]]
                writer.writerow(row)
        f.close()

def save_csv_muliple(head, input_dic, fname):
        
        f = open(fname, 'w')
        writer = csv.writer(f)
        writer.writerow([head] + list(input_dic.keys()))    
        keys = input_dic[list(input_dic.keys())[0]].keys()
        
        for k in keys:
            row = [ k]
            for query in input_dic:

                d = input_dic[query]
                row += [ d[k]]

            writer.writerow(row)

        f.close()

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d = s
        print(d)
    return d


def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def get_predecessos_nodes(event_id, graph):
    
    predecessors0 = nx.bfs_tree(graph, event_id, reverse=True)

    predecessors = list(predecessors0)

    return predecessors

def openai_imagenet_template():

    return [
        lambda c: f'a bad photo of a {c}.',
        lambda c: f'a photo of many {c}.',
        lambda c: f'a sculpture of a {c}.',
        lambda c: f'a photo of the hard to see {c}.',
        lambda c: f'a low resolution photo of the {c}.',
        lambda c: f'a rendering of a {c}.',
        lambda c: f'graffiti of a {c}.',
        lambda c: f'a bad photo of the {c}.',
        lambda c: f'a cropped photo of the {c}.',
        lambda c: f'a tattoo of a {c}.',
        lambda c: f'the embroidered {c}.',
        lambda c: f'a photo of a hard to see {c}.',
        lambda c: f'a bright photo of a {c}.',
        lambda c: f'a photo of a clean {c}.',
        lambda c: f'a photo of a dirty {c}.',
        lambda c: f'a dark photo of the {c}.',
        lambda c: f'a drawing of a {c}.',
        lambda c: f'a photo of my {c}.',
        lambda c: f'the plastic {c}.',
        lambda c: f'a photo of the cool {c}.',
        lambda c: f'a close-up photo of a {c}.',
        lambda c: f'a black and white photo of the {c}.',
        lambda c: f'a painting of the {c}.',
        lambda c: f'a painting of a {c}.',
        lambda c: f'a pixelated photo of the {c}.',
        lambda c: f'a sculpture of the {c}.',
        lambda c: f'a bright photo of the {c}.',
        lambda c: f'a cropped photo of a {c}.',
        lambda c: f'a plastic {c}.',
        lambda c: f'a photo of the dirty {c}.',
        lambda c: f'a jpeg corrupted photo of a {c}.',
        lambda c: f'a blurry photo of the {c}.',
        lambda c: f'a photo of the {c}.',
        lambda c: f'a good photo of the {c}.',
        lambda c: f'a rendering of the {c}.',
        lambda c: f'a {c} in a video game.',
        lambda c: f'a photo of one {c}.',
        lambda c: f'a doodle of a {c}.',
        lambda c: f'a close-up photo of the {c}.',
        lambda c: f'a photo of a {c}.',
        lambda c: f'the origami {c}.',
        lambda c: f'the {c} in a video game.',
        lambda c: f'a sketch of a {c}.',
        lambda c: f'a doodle of the {c}.',
        lambda c: f'a origami {c}.',
        lambda c: f'a low resolution photo of a {c}.',
        lambda c: f'the toy {c}.',
        lambda c: f'a rendition of the {c}.',
        lambda c: f'a photo of the clean {c}.',
        lambda c: f'a photo of a large {c}.',
        lambda c: f'a rendition of a {c}.',
        lambda c: f'a photo of a nice {c}.',
        lambda c: f'a photo of a weird {c}.',
        lambda c: f'a blurry photo of a {c}.',
        lambda c: f'a cartoon {c}.',
        lambda c: f'art of a {c}.',
        lambda c: f'a sketch of the {c}.',
        lambda c: f'a embroidered {c}.',
        lambda c: f'a pixelated photo of a {c}.',
        lambda c: f'itap of the {c}.',
        lambda c: f'a jpeg corrupted photo of the {c}.',
        lambda c: f'a good photo of a {c}.',
        lambda c: f'a plushie {c}.',
        lambda c: f'a photo of the nice {c}.',
        lambda c: f'a photo of the small {c}.',
        lambda c: f'a photo of the weird {c}.',
        lambda c: f'the cartoon {c}.',
        lambda c: f'art of the {c}.',
        lambda c: f'a drawing of the {c}.',
        lambda c: f'a photo of the large {c}.',
        lambda c: f'a black and white photo of a {c}.',
        lambda c: f'the plushie {c}.',
        lambda c: f'a dark photo of a {c}.',
        lambda c: f'itap of a {c}.',
        lambda c: f'graffiti of the {c}.',
        lambda c: f'a toy {c}.',
        lambda c: f'itap of my {c}.',
        lambda c: f'a photo of a cool {c}.',
        lambda c: f'a photo of a small {c}.',
        lambda c: f'a tattoo of the {c}.',
    ]

