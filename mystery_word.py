# Solving Riddler Classic @ https://fivethirtyeight.com/features/can-you-guess-the-mystery-word/
# Using Python 3.9
#
# Rather than exploring all possible word combinations, it uses a heuristic approach,
# namely: considering only words made up of the letters that are most frequent in 5-letter words,
# position by position.
# The letter frequencies and the results are visualized here: 
# https://docs.google.com/spreadsheets/d/1uosbJ1UzX5H6jZNDmuRd_lmfuK8H4QtwN_M2_soTNgI/edit?usp=sharing
#

from collections import OrderedDict
import itertools
import numpy as np
from multiset import Multiset
from pprint import pprint


N = 5
letters = 'abcdefghijklmnopqrstuvwxyz'
letters_set = set(letters)
L = len(letters)

with open('norvig_dot_com_slash_ngrams_slash_enable1.txt', 'r') as file:
    words = [line[:N] for line in file if len(line) == (N + 1)]

# # DEBUG TEST
# for w in ['saaes', 'corae', 'beity', 'pioid']:
#     if w not in words:
#         words.append(w)

MATCH_RANGE = (
    0,  # letter is NOT contained at position n
    1,  # letter IS containted at position n
    2,  # letter is contained n times at some position(s)
)

word_sets = {}
word_msets = {}
letter_lists = [[{le: [] for le in letters} for n in range(N + 1)] for match in MATCH_RANGE]
for wi, w in enumerate(words):
    word_s = set(w)
    word_sets[w] = word_s
    word_ms = Multiset(w)
    word_msets[w] = word_ms
    for n, le1 in enumerate(w):
        for le0 in letters_set.difference([le1]):
            letter_lists[0][n][le0].append(wi)
        letter_lists[1][n][le1].append(wi)
    for le2, n in word_ms.items():
        letter_lists[2][n][le2].append(wi)
    for le2 in letters_set.difference(word_s):
        letter_lists[2][0][le2].append(wi)

letter_sets = [[{le: set(letter_lists[match][n][le]) for le in letters} for n in range(N + 1)] for match in MATCH_RANGE]
n_words = len(words)
letter_sets_len = \
    [[{le: len(letter_sets[match][n][le]) for le in letters} for n in range(N + 1)] for match in MATCH_RANGE]


freqs = {(n, le): 0 for n in range(-1, N) for le in letters}
for w, wd in zip(words, word_msets):
    for le in wd:
        freqs[(-1, le)] += 1
    for n in range(N):
        freqs[(n, w[n])] += 1

for n in range(-1, N):
    for lei in range(L):
        freqs[(n, letters[lei])] /= n_words

print(','.join(['letter'] + [str(n) for n in range(-1, N)]))
for lei in range(26):
    print(','.join([letters[lei]] + [str(freqs[(n, letters[lei])]) for n in range(-1, N)]))

possible_letters_base = []
min_choices = N
threshold_ratio = 1.1
n_show = 10
for n in range(-1, N):
    print()
    print(n)
    letter2freq = {letters[lei]: freqs[(n, letters[lei])] for lei in range(L)}
    sorted_letters = sorted(letter2freq, key=letter2freq.get, reverse=True)
    for k in sorted_letters[:n_show]:
        print(f'{k},{letter2freq[k]}')
    if n >= 0:
        choices = sorted_letters[:min_choices]
        last_f = letter2freq[choices[-1]]
        for le in sorted_letters[min_choices:]:
            f = letter2freq[le]
            # print(le, last_f, f, last_f / f)
            if last_f / f >= threshold_ratio:
                break
            choices.append(le)
            last_f = f
        possible_letters_base.append(OrderedDict([(c, letter2freq[c]) for c in choices]))

print()
for od in possible_letters_base:
    print(od)

combs = list(itertools.product(
    *[list(itertools.combinations(ld.keys(), N - 1)) for ld in possible_letters_base]
))

# for comb in combs:
#     print(comb)

all_possible_letters_choices = {
    comb: np.prod([sum([possible_letters_base[i][c] for c in lc]) for i, lc in enumerate(comb)]) for comb in combs
}
all_possible_letters_choices = OrderedDict(sorted(all_possible_letters_choices.items(), key=lambda item: -item[1]))

cut_factor = 0.9
cut_threshold = cut_factor * next(iter(all_possible_letters_choices.values()))
possible_letter_choices = OrderedDict()
for plc, val in all_possible_letters_choices.items():
    if val < cut_threshold:
        break
    possible_letter_choices[plc] = all_possible_letters_choices[plc]

# for k, v in possible_letter_choices.items():
#     print(v, k)
# exit(0)

choices = OrderedDict()
for possible_letters, val in possible_letter_choices.items():

    print()
    pprint(possible_letters)
    # print(np.prod([len(c) for c in possible_letters]))

    candidates = []
    for word in words:
        if all([le in ch for le, ch in zip(word, possible_letters)]):
            candidates.append(word)
    # print(candidates)
    n_candidates = len(candidates)
    print(n_candidates)

    couples = []
    compatibility_matrix = np.ones((n_candidates, n_candidates), dtype=bool)
    for i0, c0 in enumerate(candidates):
        i1_start = i0 + 1
        for i1, c1 in enumerate(candidates[i1_start:], i1_start):
            compatible = True
            for l0, l1 in zip(c0, c1):
                if l0 == l1:
                    compatible = False
                    break
            compatibility_matrix[i0, i1] = compatibility_matrix[i1, i0] = compatible
            if compatible:
                couples.append([i0, i1])
    # print(compatibility_matrix)
    # for (c0, c1) in couples:
    #     print(candidates[c0], candidates[c1])
    n_couples = len(couples)
    print(n_couples)

    n_choices_added = 0
    for i0, (c00, c01) in enumerate(couples):
        i1_start = i0 + 1
        for i1, (c10, c11) in enumerate(couples[i1_start:], i1_start):
            if c01 < c10:
                compatible = True
                for (c0, c1) in itertools.product((c00, c01), (c10, c11)):
                    if not compatibility_matrix[c0, c1]:
                        compatible = False
                        break
                if compatible:
                    choice = tuple(candidates[c] for c in (c00, c01, c10, c11))
                    choices[choice] = val
                    n_choices_added += 1
                    # print(choice)
    print(n_choices_added)

pprint(choices)
n_choices = len(choices)
print(n_choices)


def deduce1(target, explorations):

    target_s = word_sets[target]
    target_ms = word_msets[target]
    found_letters = ['' for _ in range(N)]
    excluded_letters = [[] for _ in range(N)]
    letters_minmax = {}

    for exploration in explorations:

        for i, t in enumerate(target):
            e = exploration[i]
            if e == t:
                found_letters[i] = e
            else:
                excluded_letters[i].append(e)

        exploration_s = word_sets[exploration]
        exploration_ms = word_msets[exploration]
        ims = target_ms.intersection(exploration_ms)

        for le, ims_le in ims.items():
            if le not in letters_minmax:
                letters_minmax[le] = [ims_le, N]
            else:
                letters_minmax[le][0] = max(letters_minmax[le][0], ims_le)
            if exploration_ms[le] > ims_le:
                letters_minmax[le][1] = min(letters_minmax[le][1], ims_le)
        for le in exploration_s.difference(target_s):
            letters_minmax[le] = [0, 0]

    found_letters_str = ''.join(found_letters)
    n_unknowns = N - len(found_letters_str)
    found_letters_ms = Multiset(found_letters_str)
    for le, (_, n_max) in letters_minmax.items():
        # print(le, n_max, n_unknowns + (found_letters_ms[le] if le in found_letters_ms else 0))
        letters_minmax[le][1] = min(n_max, n_unknowns + (found_letters_ms[le] if le in found_letters_ms else 0))

    return found_letters, excluded_letters, letters_minmax


def deduce2(found_letters, excluded_letters, letters_minmax, solve=False):
    sets = []
    set_lens = []

    for n, fl in enumerate(found_letters):
        if fl:
            sets.append(letter_sets[1][n][fl])
            set_lens.append(letter_sets_len[1][n][fl])

    for n, els in enumerate(excluded_letters):
        if els:
            for el in els:
                sets.append(letter_sets[0][n][el])
                set_lens.append(letter_sets_len[0][n][el])

    ordered_sets = [sets[i] for i in np.argsort(set_lens)]
    guess_set = ordered_sets[0]
    for s in ordered_sets[1:]:
        guess_set = guess_set.intersection(s)
    possible_words = [words[w] for w in guess_set]

    final_possible_words = []
    for word in possible_words:
        word_ms = word_msets[word]
        word_ok = True
        for le, (n_min, n_max) in letters_minmax.items():
            le_n = word_ms[le] if le in word_ms else 0
            if not (n_min <= le_n <= n_max):
                word_ok = False
                break
        if word_ok:
            final_possible_words.append(word)
    retvals = [len(final_possible_words)]
    if solve:
        retvals.append(sorted(final_possible_words))
    return retvals


def deduce(target, explorations, solve=False):
    return deduce2(*deduce1(target, explorations), solve)


def guess_prob(explorations):
    probability = 0.0
    for target in words:
        probability += (1 / deduce(target, explorations)[0] / n_words)
    return probability


guess_probabilities = []
for ci, choice in enumerate(choices):
    probability = guess_prob(choice)
    print(f'choice {ci} / {n_choices}: {choice} => guess probability = {probability}')
    guess_probabilities.append(probability)
sorted_i = list(reversed(np.argsort(guess_probabilities)))
n_top_choices = 10
print(f'Best choices for the first {N - 1} words:')
for i in sorted_i[:n_top_choices]:
    print(choices[i], 'win probability =', guess_probabilities[i])
