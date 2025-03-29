def closest(word_bank: list[tuple[str, str]], target: tuple[str, str]) -> list[tuple[str, str]]:
    return sorted(word_bank, key=lambda x: closeness(x, target))[:5]


def closeness(word1: tuple[str, str], word2: tuple[str, str]) -> int:
    return len(set(word1[0] + word1[1]).intersection(set(word2[0] + word2[1])))