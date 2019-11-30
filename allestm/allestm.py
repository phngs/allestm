import sys
from argparse import ArgumentParser
import json
import pathlib
import pandas as pd
import joblib
import multiprocessing
import keras

import numpy as np

import allestm.parsers
import allestm.utils
import xgboost as xgb

from allestm.features.categorical import Sequence
from allestm.features.continuous import Pssm

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()


def parse_args(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-f', '--fasta', help='Single sequence .fasta file.')
    parser.add_argument('-a', '--a3m', help='MSA in a3m format.')
    parser.add_argument('-c', '--config', help='Config file.')
    return parser.parse_args(argv)

def prepare_data(seq, pssm, window=0):
    feature_dfs = []
    for feature, result in [(Sequence(), seq), (Pssm(), pssm)]:
        log.info(f'Calculating feature {type(feature)}.')

        df = pd.DataFrame(result, columns=[f"{type(feature).__qualname__}_{i}" for i in range(result.shape[1])]) 
        if window != 0:
            for column in df.columns:
                for win in range(-int((window - 1) / 2), int((window + 1) / 2)):
                    if win != 0:
                        new_column = f"{column}_w{win}"
                        df[new_column] = df[column].shift(win)

        feature_dfs.append(df.ffill().bfill())

    return feature_dfs


def main():
    args = parse_args(sys.argv[1:])

    # Parsing config.
    log.debug('Parsing config file.')
    with open(args.config, 'r') as config_fh:
        config = json.load(config_fh)

    # Parsing fasta file.
    log.debug('Parsing fasta.')
    fasta = allestm.parsers.parse_fasta(args.fasta)
    seq_transformer = allestm.features.categorical.Sequence()
    seq = seq_transformer.transform(fasta)

    # Parsing a3m file.
    log.debug('Parsing a3m.')
    a3m = allestm.parsers.parse_a3m(args.a3m)
    pssm_transformer = allestm.features.continuous.Pssm()
    pssm = pssm_transformer.transform(a3m)

    # Cache: cache[FOLD][METHOD][TARGET] = PREDICTIONS.
    results = {}

    do_dl = False
    do_rf = False
    do_xgb = False
    do_blending = True
    do_avg = True

    # DL.
    if do_dl:
        def run_dl(seq, pssm, model_file, queue):
            model = keras.models.load_model(model_file)
            predictions = model.predict([dl_seq_data, dl_pssm_data])
            queue.put(predictions)

        dl_seq_data_tmp, dl_pssm_data_tmp = (x.to_numpy() for x in prepare_data(seq, pssm, window=0))

        dl_seq_data = np.zeros((1, config['config']['max_length'], len(seq_transformer)))
        dl_pssm_data = np.zeros((1, config['config']['max_length'], len(seq_transformer)))

        dl_seq_data[0, :dl_seq_data_tmp.shape[0], :dl_seq_data_tmp.shape[1]] = dl_seq_data_tmp
        dl_pssm_data[0, :dl_pssm_data_tmp.shape[0], :dl_pssm_data_tmp.shape[1]] = dl_pssm_data_tmp

        for method in ['cnn', 'dcnn', 'lstm']:
            for target_group in config['methods'][method]:
                log.debug(f'Target group {target_group}.')

                for model_config in config['methods'][method][target_group]:
                    log.debug(f'Model config: {model_config}.')
                    queue = multiprocessing.SimpleQueue()

                    model_file = f'models/{model_config["model_file"]}'
                    log.debug(f'Loading model file {model_file} for targets {model_config["targets"]}.')

                    # Necessary to free memory after prediction.
                    p = multiprocessing.Process(target=run_dl, args=(dl_seq_data, dl_pssm_data, model_file, queue))
                    p.start()
                    p.join()

                    predictions = queue.get()
                    for i, target_name in enumerate(model_config["targets"].split(',')):
                        target = allestm.utils.name_to_feature(f'allestm.features.{target_name}')

                        if len(predictions[i].shape) > 2:
                            preds = predictions[i][0, :dl_seq_data_tmp.shape[0], :]
                        else:
                            preds = predictions[i][:dl_seq_data_tmp.shape[0], :]

                        if predictions[i].shape[-1] == 1:
                            preds = preds.squeeze()

                        if hasattr(target, 'inverse_transform'):
                            preds = target.inverse_transform(preds)

                        results.setdefault(target_name, {}).setdefault(method, {}).setdefault(model_config['fold'], preds.tolist())

    # RF.
    if do_rf:
        for target_group in config['methods']['rf']:
            log.debug(f'Target group {target_group}.')

            for model_config in config['methods']['rf'][target_group]:
                log.debug(f'Model config: {model_config}.')
                data = pd.concat(prepare_data(seq, pssm, window=model_config['window_size']), axis=1)

                target = allestm.utils.name_to_feature(f'allestm.features.{model_config["targets"]}')

                model_file = f'models/{model_config["model_file"]}'
                log.debug(f'Loading model file {model_file} for target {target}.')
                model = joblib.load(model_file)

                if isinstance(model, RandomForestRegressor):
                    predictions = model.predict(data)
                else:
                    predictions = model.predict_proba(data)

                if isinstance(target, allestm.features.binary.Binary):
                    predictions = predictions[:, 1]

                if hasattr(target, 'inverse_transform'):
                    predictions = target.inverse_transform(predictions)

                results.setdefault(model_config['targets'], {}).setdefault('rf', {}).setdefault(model_config['fold'], predictions.tolist())

    # XGB.
    if do_xgb:
        for target_group in config['methods']['xgb']:
            log.debug(f'Target group {target_group}.')

            for model_config in config['methods']['xgb'][target_group]:
                log.debug(f'Model config: {model_config}.')
                data = pd.concat(prepare_data(seq, pssm, window=model_config['window_size']), axis=1)

                target = allestm.utils.name_to_feature(f'allestm.features.{model_config["targets"]}')

                model_file = f'models/{model_config["model_file"]}'
                log.debug(f'Loading model file {model_file} for target {target}.')

                model = xgb.Booster({'nthread': multiprocessing.cpu_count()})
                model.load_model(model_file)

                predictions = model.predict(xgb.DMatrix(data), ntree_limit=int(model.attributes()['best_iteration']) + 1)

                if hasattr(target, 'inverse_transform'):
                    predictions = target.inverse_transform(predictions)

                results.setdefault(model_config['targets'], {}).setdefault('xgb', {}).setdefault(model_config['fold'], predictions.tolist())

    with open('tmp.json', 'r') as fh:
        results = json.load(fh)

# ["lstm_", "cnn_", "dcnn_", "rf_", "xgb_"]
# * 5 folds
# blending
# avg
# x targets



    # Blending
    if do_blending:
        for target_group in config['methods']['blending']:
            log.debug(f'Target group {target_group}.')

            for model_config in config['methods']['blending'][target_group]:
                log.debug(f'Model config: {model_config}.')
                data = np.column_stack([np.array(results[model_config['targets']][method][str(model_config['fold'])]) for method in ['cnn', 'dcnn', 'lstm', 'rf', 'xgb']])

                target = allestm.utils.name_to_feature(f'allestm.features.{model_config["targets"]}')

                model_file = f'models/{model_config["model_file"]}'
                log.debug(f'Loading model file {model_file} for target {target}.')
                model = joblib.load(model_file)

                if isinstance(model, LinearRegression):
                    predictions = model.predict(data)
                else:
                    predictions = model.predict_proba(data)

                results.setdefault(model_config['targets'], {}).setdefault('blending', {}).setdefault(model_config['fold'], predictions.tolist())

    # Avg
    if do_avg:
        for target in results:
            avg = np.sum([results[target]['blending'][x] for x in results[target]['blending']], axis=0)
            avg /= len(results[target]['blending'])
            results.setdefault(target, {}).setdefault('avg', avg.tolist())

    print(results)

if __name__ == '__main__':
    main()
