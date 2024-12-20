import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import trange

from interplm.concept.uniprotkb_concept_constants import \
    subconcepts_to_exclude_from_evals


def make_eval_subset(shards_to_include: List[int],
                     uniprot_dir: Path,
                     eval_name: str = "eval",
                     min_aa_per_concept: int = 1500,
                     min_domains_per_concept: int = 10):
    """
    Create a subset of the UniprotKB amino acid concept data for evaluation.

    Each concept must have at least `min_aa_per_concept` amino acids OR `min_domains_per_concept`
    domains to be included in the evaluation set.

    Args:
        shards_to_include: 2 element list indicating start (inclusive) and end (exclusive) inices
            of shards to include
        uniprot_dir: Path to the UniprotKB directory
        min_aa_per_concept: Minimum number of amino acids per concept to include
        min_domains_per_concept: Minimum number of domains per concept to include
    """

    shards_to_include = list(range(shards_to_include[0], shards_to_include[1]))

    n_positive_aa_per_concept = []
    n_positive_domains_per_concept = []

    # Calculate the number of positive amino acids per concept and the number of positive domains per concept
    n_amino_acids = 0
    for i in shards_to_include:
        res = sparse.load_npz(uniprot_dir / f"shard_{i}/aa_concepts.npz").toarray()
        n_positive_aa_per_concept.append(np.count_nonzero(res, axis=0).tolist())
        n_positive_domains_per_concept.append(res.max(axis=0).tolist())
        n_amino_acids += len(res)

    with open(uniprot_dir / "uniprotkb_aa_concepts_columns.txt") as f:
        all_concept_names = f.read().splitlines()

    # import pdb; pdb.set_trace()

    indices_of_many_domains = np.where(np.array(n_positive_domains_per_concept).sum(axis=0) > min_domains_per_concept)[0]
    indices_of_many_aa = np.where(np.array(n_positive_aa_per_concept).sum(axis=0) > min_aa_per_concept)[0]

    # Some of the catch-all sub-concepts are not meaningful for evaluation, so we exclude them
    # (e.g. "any Region" is not interesting whereas "any Zinc Finger" is)
    indices_of_concepts_to_ignore = [all_concept_names.index(c) for c in subconcepts_to_exclude_from_evals]

    concept_idx_to_keep = (set(indices_of_many_domains) | set(indices_of_many_aa)) - set(indices_of_concepts_to_ignore)
    concept_idx_to_keep = sorted([int(i) for i in concept_idx_to_keep])

    # Make the evaluation directory
    test_dir = uniprot_dir / eval_name
    test_dir.mkdir(parents=True, exist_ok=True)

    full_paths_per_shard = {i: str((uniprot_dir / f"shard_{i}" / "aa_concepts.npz").resolve()) for i in shards_to_include}

    with open(test_dir / "aa_concepts_columns.txt", "w") as f:
        f.write("\n".join([all_concept_names[i] for i in concept_idx_to_keep]))

    # Save the metadata for the evaluation set
    with open(test_dir / "metadata.json", "w") as f:
        metadata = {
            "n_concepts": len(concept_idx_to_keep),
            "n_amino_acids": n_amino_acids,
            "shard_source": shards_to_include,
            "n_positive_aa_per_concept": np.array(n_positive_aa_per_concept)[:, concept_idx_to_keep].tolist(),
            "n_positive_domains_per_concept": np.array(n_positive_domains_per_concept)[:, concept_idx_to_keep].tolist(),
            "indices_of_concepts_to_keep": concept_idx_to_keep,
            "path_to_shards": full_paths_per_shard
        }
        json.dump(metadata, f)

    print(f"Concept evaluation subset created in {test_dir}")


if __name__ == "__main__":
    from tap import tapify
    tapify(make_eval_subset)
