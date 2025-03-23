""" DOCSTRING... """

import csv
from typing import Any, Dict, Tuple

import numpy as np
from numpy import dtype, ndarray

VECTOR_IDX_HIGH = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 19, 23, 24])
ID_IDX_HIGH = np.array([17, 18])


def generate_tables(file: str = "high_popularity_spotify_data.csv", id_idx: np.ndarray[Any, dtype] = ID_IDX_HIGH,
                    vector_idx: np.ndarray[Any, dtype] = VECTOR_IDX_HIGH) -> tuple[
    ndarray[Any, dtype[Any]], dict[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]], dict[
        ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]], ndarray[Any, dtype[Any]]]:
    """
    Return the id list, feature vector dictionary, the corresponding name look up table to map each song
    id in the dataset to the song name and to its feature vector.

    Preconditions:
        - file is a valid csv_file
        - all values of the index lists are valid indices of the file
    """

    data = []
    with open(file, 'r') as file:
        lines = csv.reader(file)

        for line in lines:
            data.append(line)
    data = np.array(data)

    vector_labels = data[0, vector_idx]
    vectors = data[1:, vector_idx].astype(float)
    identifiers = data[1:, id_idx]

    vector_table = dict()
    name_table = dict()
    for i in range(0, vectors.shape[0]):
        vector_table[identifiers[i, 0]] = vectors[i, :]
        name_table[identifiers[i, 0]] = identifiers[i, 1]
    return (identifiers[:, 0], vector_table, name_table, vector_labels)


def parse_file(file: str = "high_popularity_spotify_data.csv", id_idx: np.ndarray[Any, dtype] = ID_IDX_HIGH,
               vector_idx: np.ndarray[Any, dtype] = VECTOR_IDX_HIGH) -> tuple[
               ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """ Parse file and separate labels, feature vectors and ids into respective arrays
    """
    data = []
    with open(file, 'r') as file:
        lines = csv.reader(file)

        for line in lines:
            data.append(line)
    data = np.array(data)
    return data[0, vector_idx], data[1:, vector_idx].astype(float), data[1:, id_idx]


def normalize(feature_vectors: np.ndarray[Any, dtype]) -> tuple[
        ndarray[dtype[Any, dtype]], ndarray[Any, dtype], ndarray[Any, dtype]]:
    """ Normalize feature vectors along each dimension between 0 and 1 and return the mins and maxes
    for these dimensions"""

    mins = feature_vectors.min(axis=0)
    maxes = feature_vectors.max(axis=0)
    normed_vectors = (feature_vectors - mins) / (maxes-mins)

    return normed_vectors, mins, maxes


def build_tables(normed_vectors: np.ndarray[Any, dtype], mins: np.ndarray[Any, dtype], maxes: np.ndarray[Any, dtype],
                 identifiers: np.ndarray[Any, dtype]) -> tuple[
    ndarray[Any, dtype], dict[ndarray[Any, dtype], ndarray[Any, dtype]], dict[
        ndarray[Any, dtype], tuple[ndarray[Any, dtype], ndarray[Any, dtype]]], dict[
        ndarray[Any, dtype], ndarray[Any, dtype]]]:
    """ Build look up tables for each vector and id array
    """

    vector_table = dict()
    norm_table = dict()
    name_table = dict()
    for i in range(0, normed_vectors.shape[0]):
        vector_table[identifiers[i, 0]] = normed_vectors[i, :]
        name_table[identifiers[i, 0]] = identifiers[i, 1]
        norm_table[identifiers[i, 0]] = (mins[i], maxes[i])

    return (identifiers[:, 0], vector_table, norm_table, name_table)
