""" Module containing functions for matching similar words"""

def closest(word_bank: list[tuple[str, str]], target: tuple[str, str]) -> list[tuple[str, str]]:
    """
    Return 5 closest words in the word bank to the given target word
    """
    return sorted(word_bank, key=lambda x: closeness(x, target), reverse=True)[:5]


def closeness(word1: tuple[str, str], word2: tuple[str, str]) -> float:
    """
    Return the similarity between two words, calculated through a mixture of word based and character based
    similarity
    """
    token1 = (word1[0] + " " +  word1[1]).lower()
    token2 = (word2[0] + " " +  word2[1]).lower()

    s1, s2 = set(token1), set(token2)
    char_sim = len(s1 & s2) / max(len(s1 | s2), 1)

    s1, s2 = set(token1.split()), set(token2.split())
    word_sim = len(s1 & s2) / max(len(s1 | s2), 1)

    return 0.5 * char_sim + 0.5 * word_sim
