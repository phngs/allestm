import sqlite3
import tempfile
import json
import pathlib

import numpy as np

from mllib.retrievers import SQLRetriever


def db_store_predicted(conn, experiment, target, ids, predicted, simulate=False):
    """
    Store the observed and predicted values in the database.

    Args:
        conn: Sqlite3 connection object.
        experiment: Name of the experiement to be stored in the db.
        target: Target feature instance.
        ids: List of ids to be stored.
        predicted: Predicted values.
        simulate: If true, the insert is only simulated.

    Returns:
        Nothing.
    """
    cur = conn.cursor()

    table_name = "{}.{}".format(type(target).__module__, type(target).__qualname__).replace(".", "_")
    columns_def = ", ".join(["val_{} FLOAT".format(x) for x in range(len(target))])
    columns = ", ".join(["val_{}".format(x) for x in range(len(target))])
    placeholders = ", ".join(["?" for _ in range(len(target))])

    create_query = "CREATE TABLE IF NOT EXISTS {} (experiment VARCHAR(50), id VARCHAR(6), resi INT, {}, PRIMARY KEY(experiment, id, resi))".format(
        table_name, columns_def)
    if simulate:
        print(create_query)
    else:
        cur.execute(create_query)

    insert_query = "INSERT OR REPLACE INTO {} (experiment, id, resi, {}) VALUES (?, ?, ?, {})".format(table_name,
                                                                                                      columns,
                                                                                                      placeholders)
    params = []
    for i, protein_pred in enumerate(predicted):
        for resi, pred in enumerate(protein_pred):
            params.append((experiment, ids[i], resi, *pred.tolist()))

    # for param in params:
    # if simulate:
    # print(insert_query, param)
    # else:
    # cur.execute(insert_query, param)

    if simulate:
        print(insert_query, params)
    else:
        cur.executemany(insert_query, params)

    conn.commit()


def db_create_experiments_table(conn):
    """
    Create experiments table.

    Args:
        conn: Sqlite3 connection object.

    Returns:
        Nothing.
    """
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS experiments (experiment VARCHAR(50), fold INT, path TEXT, history TEXT, parameters TEXT, loss FLOAT, primary key (experiment, fold))")

    conn.commit()


def db_model_exists(conn, experiment, fold):
    """
    Check if model already exists in database.

    Args:
        conn: Sqlite3 connection object.
        experiment: Experiment name.
        fold: Fold number.

    Returns:
        True if model exists, False otherwise.
    """
    cur = conn.cursor()

    for _ in cur.execute("SELECT * FROM experiments WHERE experiment=? and fold=?", (experiment, fold)):
        return True

    return False


def db_get_loss(conn, experiment, fold):
    """
    Get loss of a model in db.

    Args:
        conn: Sqlite3 connection object.
        experiment: Experiment name.
        fold: Fold number.

    Returns:
        The loss of the model.
    """
    cur = conn.cursor()

    loss = cur.execute("SELECT loss FROM experiments WHERE experiment=? and fold=?", (experiment, fold)).fetchone()[0]

    return loss


def get_num_folds(conn, dataset_name):
    """
    Get the number of folds for a given dataset_name.
    Basically returns only the highest fold number + 1, i.e. if only fold 9
    exists (but not 1, 2, 3, ..., 8), returns 9 anyway.

    Args:
        conn: Sqlite3 connection object.
        dataset_name: Name of the dataset.

    Returns:
        Number of folds.
    """
    ret = SQLRetriever(conn, "SELECT MAX(fold) FROM datasets WHERE name=?")
    return ret.transform(dataset_name)[0, 0] + 1


def get_ids_and_lengths(conn, dataset_name, kind, fold, limit=None):
    """
    Get the ids and the lengths for a given dataset_name, kind and fold.

    Args:
        conn: Sqlite3 connection object.
        dataset_name: Name of the dataset.
        kind: Kind of the dataset (train, valid, test).
        fold: Number of the fold retrieved from the DB.

    Returns:
        (number_of_entries, 2)-shaped list of lists.
    """
    query = "SELECT distinct id, length FROM datasets JOIN proteins using (id) WHERE name=? AND kind=? AND fold=?"
    if limit:
        query = "{} limit {}".format(query, limit)
    ret = SQLRetriever(conn, query)
    array = ret.transform(dataset_name, kind, fold).transpose()
    return [array[0].tolist(), [int(x) for x in array[1].tolist()]]


def get_max_length(conn, dataset_name, fold):
    """
    Get the maximum sequence length for a given dataset_name and fold.

    Args:
        conn: Sqlite3 connection object.
        dataset_name: Name of the dataset.
        fold: Number of the fold.

    Returns:
        Maximum sequence length.
    """
    ret = SQLRetriever(conn, "SELECT MAX(length) FROM datasets JOIN proteins USING (id) WHERE name=? AND fold=?")
    return ret.transform(dataset_name, fold)[0, 0]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def db_store_model(conn, db_file, save_func, experiment, fold, model, history, parameters, loss):
    """
    Store the model for an experiment and fold into the database.

    Args:
        conn: Sqlite3 connection object.
        db_file: Path to the sqlite3 db to create model folder.
        save_func: Function taking the model and a filename as arguments to save the model to file.
        experiment: Experiment name.
        fold: Fold number.
        model: Model.
        history: History dict.
        parameters: Parameters dict.
        loss: Loss.

    Returns:
        Nothing.
    """
    cur = conn.cursor()

    path = pathlib.Path(db_file).parent / 'models'
    path.mkdir(parents=True, exist_ok=True)

    model_file = path / f'{experiment}_{fold}.model'

    save_func(model, model_file)

    cur.execute(
        "INSERT OR REPLACE INTO experiments (experiment, fold, path, history, parameters, loss) VALUES (?, ?, ?, ?, ?, ?)",
        (experiment, fold, str(model_file), json.dumps(history, cls=NumpyEncoder), json.dumps(parameters, cls=NumpyEncoder), loss))

    conn.commit()
