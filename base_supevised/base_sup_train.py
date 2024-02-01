from sklearn import svm
import pickle
import pandas as pd
import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, root)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from base_sup_utils import *
from base_sup_args import get_parser_sup
from base_sup_utils import *
import torch
args = get_parser_sup()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_model( x_val, y_val):
    model = None
    best_params = None

    if args.model_type == 'svm':

        model = svm.SVC()
        param_grid = {'C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000] }
        grid_search = GridSearchCV( model, param_grid, scoring=['accuracy'] , refit="accuracy" )
        grid_search.fit(x_val, y_val)
        best_params = grid_search.best_params_
        model = svm.SVC(C=best_params["C"], gamma=best_params["gamma"], kernel=best_params["kernel"] )

    elif args.model_type in ['linear', 'reg']:
        model = LogisticRegression(random_state=0, max_iter=1000, verbose=1, n_jobs=4)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 1000]}
        grid_search = GridSearchCV( model, param_grid, scoring=['accuracy'] , refit="accuracy" )
        grid_search.fit(x_val, y_val)
        best_params = grid_search.best_params_
        model = LogisticRegression(random_state=0, C=best_params['C'], max_iter=1000, verbose=1, n_jobs=4)
        
    return model, best_params

for dataset in ['instances', 'vise_bing', 'red', 'wider', 'soceid']:
    args.dataset = dataset
    if 'vise' in args.dataset : chk_dataset = 'vise'
    else: chk_dataset = args.dataset

    for n_shot in [1,2,3,4,5,10,20,30,50]:
        print('***', n_shot)

        for n_round in [1, 2, 3]:
            print('** round', n_round)

            if os.path.exists(f'{args.checkpoints_path}/{args.model_type}/{args.model_pretrained}_{args.model_type}_{chk_dataset}_shot_{n_shot}_round_{n_round}.pickle'):
                continue

            ## read data
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
            x_train, y_train, _ = get_image_embeddings(caption_chk_path= args.caption_chk_path, image_h5_path = train_path_h5, csv_path = train_data_files, valid_ids = train_valid_ids, data_split = 'train', n_round= n_round )
            print('train images loaded')
            x_val, y_val, _ = get_image_embeddings(caption_chk_path=args.caption_chk_path , image_h5_path = val_path_h5, csv_path = val_data_files, valid_ids = val_valid_ids, data_split = 'val', n_round= n_round)
            print('images loaded')

            # hyper parameter tuning
            num_class = len(list( set(list (y_train))))
            
            print('hyperparameter tuning')
            model, best_params = get_best_model( x_val, y_val)
            
            print('model training')
            model.fit(x_train, y_train)

            info = {'model': args.model_type, 'dataset':chk_dataset, 'params': best_params}

            # save model
            with open(f'{args.checkpoints_path}/{args.model_type}/{args.model_pretrained}_{args.model_type}_{chk_dataset}_shot_{n_shot}_round_{n_round}.pickle', 'wb') as f:
                pickle.dump(model, f)

            # save hyperparameter info
            # with open(f'{args.path_results}/{args.model_type}/info_{args.model_pretrained}_{args.model_type}_{chk_dataset}_shot_{n_shot}_round_{n_round}.json', 'w') as f:  
            #     json.dump(info, f)
