import pandas as pd
import sys
import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, root)
from base_sup_utils import *
import numpy as np
from sklearn.metrics import top_k_accuracy_score, confusion_matrix
import pickle
from base_sup_utils import *



for dataset in [ 'vise_bing', 'vise_wiki', 'instances', 'wider', 'soceid', 'instances' ]: 
    args.dataset = dataset

    for n_shot in [ 1, 2, 3, 4, 5, 10, 20, 30, 50 ]:

        META_acc_events = {}
        META_acc_total = []

        for n_round in [1,2,3]:
            print('shot', n_shot, 'round', n_round)
            
            # read data
            paths = get_data_path(n_shot=n_shot, n_round=n_round, dataset=args.dataset, data_path_=args.data_path_, model_pretrained = args.model_pretrained)
            [train_path, val_path, test_path, class_names_path, graph_path] = paths['raw']
            [train_path_h5, val_path_h5, test_path_image_h5] = paths['h5']

            train_data_files = pd.read_csv(train_path)
            val_data_files = pd.read_csv(val_path)

            # valid ids
            if 'vise' in args.dataset:
                train_valid_ids = [f.split('/')[-1].split('.')[0] for f in train_data_files['filepath'] ]
                val_valid_ids = [f.split('/')[-1].split('.')[0] for f in val_data_files['filepath'] ]
            else:
                train_valid_ids = [f.split('/')[-1].split('.')[0]+'+'+f.split('/')[-2] for f in train_data_files['filepath'] ]
                val_valid_ids = [f.split('/')[-1].split('.')[0]+'+'+f.split('/')[-2] for f in val_data_files['filepath'] ]

            class_names = pd.read_csv(class_names_path)['gt_label']
            print('data paths loaded')

            # get image embs
            x_test, y_test, _ = get_image_embeddings(caption_chk_path= args.caption_chk_path, image_h5_path = test_path_image_h5, csv_path = test_path, data_split = 'test', n_round= n_round)

            if 'vise' in args.dataset : chk_dataset = 'vise'
            else: chk_dataset = args.dataset

            # load model
            with open(f'{args.checkpoints_path}/{args.model_type}/{args.model_pretrained}_{args.model_type}_{chk_dataset}_shot_{n_shot}_round_{n_round}.pickle', 'rb') as f:
                model = pickle.load(f)

            # predict
            y_pred = model.predict(x_test)       
            
            # evaluate
            accuracy_total = []
            decision_scores = model.decision_function(x_test)
           
            for k in [1,3,5]:
                accuracy_total.append( np.round (top_k_accuracy_score(y_test, decision_scores, k=k, labels = list( class_names ) ) * 100, 2 ) )
        
            META_acc_total.append(accuracy_total)
            
            #  evaluate per event
            matrix = confusion_matrix(y_test, y_pred)
            acc_all = matrix.diagonal()/matrix.sum(axis=1)
            classes = model.classes_

            for icls in range(len(classes)):  
                cls_name = classes[icls] 
                if cls_name not in META_acc_events: META_acc_events[cls_name] = []
                META_acc_events[cls_name].append( acc_all[icls] )

        # save results
        for cls_ in META_acc_events:  META_acc_events[cls_] = np.round( np.mean( META_acc_events[cls_] ), 2 )
        META_acc_total = np.round( np.mean( META_acc_total, axis=0 ), 2 )

        META_acc_events['total'] = list( META_acc_total )
        print(META_acc_total)

        with open(f'{args.path_results}/{args.model_type}/results_{args.model_type}_{args.model_pretrained}_{args.dataset}_shot_{n_shot}.json', 'w') as f:   
            json.dump(META_acc_events, f)
