""" DOCSTRING... """

import csv
from typing import Any

import numpy as np
from numpy import dtype, ndarray
import re

VECTOR_IDX_HIGH = np.array([0, 1, 2, 4, 5, 6, 9, 19, 23, 24])
SOFT_IDX_HIGH = np.array([3, 10, 14, 26, 27])
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
               id_idx: np.ndarray[Any, dtype]) -> tuple[
               ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """ Parse file and separate labels, feature vectors and ids into respective arrays
    """
    data = []
    with open(file, 'r') as file:
        lines = csv.reader(file)
        i = 0
        for line in lines:
            # if i == 200:
            #     break
            # i += 1
            data.append(line)
    data = np.array(data)

    return data[0, vector_idx], np.vectorize(replace_non_numeric)(data[1:, vector_idx]), data[1:, id_idx]

def parse_files(file1_data: tuple[str, Any, Any] = ("high_popularity_spotify_data.csv", VECTOR_IDX_HIGH, ID_IDX_HIGH), 
                file2_data: tuple[str, Any, Any] = ("low_popularity_spotify_data.csv", VECTOR_IDX_LOW, ID_IDX_LOW)) -> tuple[
                ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    high_data = parse_file(file1_data[0], file1_data[1], file1_data[2])
    low_data = parse_file(file2_data[0], file2_data[1], file2_data[2])
    
    return high_data[0], np.concatenate((high_data[1], low_data[1]), axis=0), np.concatenate((high_data[2], low_data[2]), axis=0)


def normalize(feature_vectors: np.ndarray[Any, dtype]) -> tuple[
        ndarray[Any, dtype], ndarray[Any, dtype], ndarray[Any, dtype]]:
    """ Normalize feature vectors along each dimension between 0 and 1 and return the mins and maxes
    for these dimensions"""

    mins = feature_vectors.min(axis=0)
    maxes = feature_vectors.max(axis=0)
    normed_vectors = 2 * (feature_vectors - mins) / (maxes-mins) - 1

    return normed_vectors, mins, maxes


def build_tables(normed_vectors: np.ndarray[Any, dtype],
                 identifiers: np.ndarray[Any, dtype]) -> tuple[
    ndarray[Any, dtype], dict[ndarray[Any, dtype], ndarray[Any, dtype]], dict[
        ndarray[Any, dtype], tuple[ndarray[Any, dtype], ndarray[Any, dtype]]], dict[
        ndarray[Any, dtype], ndarray[Any, dtype]]]:
    """ Build look up tables for each vector and id array
    """

    vector_table = dict()
    name_table = dict()
    for i in range(0, normed_vectors.shape[0]):
        vector_table[identifiers[i, 0]] = normed_vectors[i, :]
        name_table[identifiers[i, 0]] = clean_text(identifiers[i, 1])

    return (identifiers[:, 0], vector_table, name_table)


