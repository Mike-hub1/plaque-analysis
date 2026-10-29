"""Animal-isolated folds and unique age-stratified construction sets."""
import math
import numpy as np


def inner_splits(groups, folds, seed):
    groups = np.asarray(groups).astype(str)
    animals = np.array(sorted(set(groups)))
    if not 2 <= folds <= len(animals):
        raise ValueError(f"inner_folds={folds} requires at least {folds} available animals")
    np.random.default_rng(seed).shuffle(animals)
    for held in np.array_split(animals, folds):
        validation = np.isin(groups, held)
        yield np.flatnonzero(~validation), np.flatnonzero(validation)


def outer_splits(groups):
    groups = np.asarray(groups).astype(str)
    for animal in sorted(set(groups)):
        test = groups == animal
        yield animal, np.flatnonzero(~test), np.flatnonzero(test)


def stratified_animal_sets(frame, config):
    animal_age = frame[[config.animal_column, config.age_column]].drop_duplicates()
    pools = [np.array(sorted(part[config.animal_column].astype(str)))
             for _, part in animal_age.groupby(config.age_column, sort=True)]
    take = config.animals_per_age
    if any(len(pool) <= take for pool in pools):
        raise ValueError("Each age must retain at least one evaluation animal after construction sampling")
    possible = math.prod(math.comb(len(pool), take) for pool in pools)
    if config.stability_repeats > possible:
        raise ValueError(f"Only {possible} distinct age-stratified construction sets are possible")
    rng, seen, sets = np.random.default_rng(config.seed), set(), []
    for _ in range(max(10000, config.stability_repeats * 1000)):
        selection = tuple(sorted(a for pool in pools for a in rng.choice(pool, take, replace=False)))
        if selection not in seen:
            sets.append(selection)
            seen.add(selection)
        if len(sets) == config.stability_repeats:
            return sets
    raise RuntimeError("Could not draw the requested number of unique animal sets")
