""" DOCSTRING... """

import csv
from typing import Any

import numpy as np
from numpy import dtype, ndarray
import re

VECTOR_IDX_HIGH = np.array([0, 1, 2, 4, 5, 6, 9, 19, 23, 24])
SOFT_IDX_HIGH = np.array([3, 10, 14, 26])
ID_IDX_HIGH = np.array([16, 17, 7])
VECTOR_IDX_LOW = np.array([7, 24, 3, 25, 27, 22, 0, 2, 1, 18, 6, 26]) #Outdated
ID_IDX_LOW = np.array([21, 11])

def clean_text(value):
    value = re.sub(r'[^a-zA-Z0-9\s]', '', value)
    value = re.sub(r'feat\.[^a-zA-Z0-9]', '', value)
    return value

def replace_non_numeric(value):
    try:
        return float(value)
    except ValueError:
        return 0.0 

def parse_file(file: str, vector_idx: np.ndarray[Any, dtype],
               id_idx: np.ndarray[Any, dtype], soft_idx: np.ndarray[Any, dtype]) -> tuple[
               ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """ Parse file and separate feature vectors and ids and soft descriptors into respective arrays
    """
    data = []
    with open(file, 'r') as file:
        lines = csv.reader(file)
        i = 0
        for line in lines:
            # if i == 200:
            #     break
            i += 1
            data.append(line)
    data = np.array(data)

    return np.vectorize(replace_non_numeric)(data[1:, vector_idx]), data[1:, id_idx], data[1:, soft_idx]

def normalize(feature_vectors: np.ndarray[Any, dtype]) -> tuple[
        ndarray[Any, dtype], ndarray[Any, dtype], ndarray[Any, dtype]]:
    """ Normalize feature vectors along each dimension between 0 and 1 and return the mins and maxes
    for these dimensions"""

    mins = feature_vectors.min(axis=0)
    maxes = feature_vectors.max(axis=0)
    normed_vectors = 2 * (feature_vectors - mins) / (maxes-mins) - 1

    return normed_vectors, mins, maxes


def build_tables(normed_vectors: np.ndarray[Any, dtype],
                 identifiers: np.ndarray[Any, dtype], soft_attributes: np.ndarray[Any, dtype]) -> tuple[Any, Any, Any, Any]:
    """ Build look up tables for data
    """

    vector_table = dict()
    name_table = dict()
    soft_table = dict()
    for i in range(0, normed_vectors.shape[0]):
        vector_table[identifiers[i, 0]] = normed_vectors[i, :]
        name_table[identifiers[i, 0]] = (clean_text(identifiers[i, 1]), clean_text(identifiers[i, 2]))
        soft_table[identifiers[i, 0]] = soft_attributes[i, :]

    return (identifiers[:, 0], vector_table, name_table, soft_attributes)


