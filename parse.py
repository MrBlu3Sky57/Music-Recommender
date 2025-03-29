""" Module containing functions that clean file data and parse into useful datastructures"""

import csv
from typing import Any
import re
import numpy as np
from numpy import dtype, ndarray

VECTOR_IDX = np.array([0, 1, 2, 4, 5, 6, 9, 19, 23, 24])
ID_IDX = np.array([16, 17, 7])
SOFT_IDX = np.array([3, 10, 14, 26])

def clean_text(value: str) -> str:
    """ Filter out unwanted characters in string and convert to lower case"""
    value = re.sub(r'[^a-zA-Z0-9\s,]', '', value)
    return str.lower(value)

def replace_non_numeric(value: str) -> float:
    """ Convert a string to float, if string cannot be converted set value to 0.0"""
    try:
        return float(value)
    except ValueError:
        return 0.0 

def parse_file(file: str = 'high_popularity_spotify_data.csv', vector_idx: np.ndarray[Any, dtype] = VECTOR_IDX,
               id_idx: np.ndarray[Any, dtype] = ID_IDX, soft_idx: np.ndarray[Any, dtype] = SOFT_IDX) -> tuple[
               ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """ Parse file and separate feature vectors and ids and soft descriptors into respective arrays
    """
    data = []
    with open(file, 'r') as file:
        lines = csv.reader(file)
        for line in lines:
            data.append(line)
    data = np.array(data, dtype=str)

    return np.vectorize(replace_non_numeric)(data[1:, vector_idx]), data[1:, id_idx], data[1:, soft_idx]

def normalize(feature_vectors: np.ndarray[Any, dtype]) -> ndarray[Any, dtype]:
    """ Normalize feature vectors along each dimension between 0 and 1 and return the mins and maxes
    for these dimensions"""

    mins = feature_vectors.min(axis=0)
    maxes = feature_vectors.max(axis=0)
    normed_vectors = 2 * (feature_vectors - mins) / (maxes-mins) - 1

    return normed_vectors


def build_tables(normed_vectors: np.ndarray[Any, dtype],
                 identifiers: np.ndarray[Any, dtype], soft_attributes: np.ndarray[Any, dtype]) -> tuple[Any, Any, Any, Any]:
    """ Build look up tables for data, and remove any duplicate song names
    """

    vector_table = dict()
    name_table = dict()
    soft_table = dict()
    seen_keys = set()
    ids = []
    for i in range(0, normed_vectors.shape[0]):
        song_name = clean_text(identifiers[i, 1])
        artist_name = clean_text(identifiers[i, 2])
        key = (song_name, artist_name)

        if key in seen_keys:
            continue
        seen_keys.add(key)
        vector_table[identifiers[i, 0]] = normed_vectors[i, :]
        name_table[identifiers[i, 0]] = key
        soft_table[identifiers[i, 0]] = soft_attributes[i, :]
        ids.append(identifiers[i, 0])

    return (np.array(ids), vector_table, name_table, soft_table)


