"""
Convert a directory of FASTA files to a directories of ESM layer activations organized
by layer and shard with specific metadata used for SAE training.
"""
import json
import os
from pathlib import Path
from typing import List, Tuple

from interplm.data_processing.utils import read_fasta

import numpy as np
import torch
from esm import FastaBatchedDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from interplm.esm.embed import get_model_converter_alphabet
    

def get_activations(
    model: torch.nn.Module,
    model_name: str,
    batch_tokens: torch.Tensor,
    batch_mask: torch.Tensor,
    layers: List[int],

) -> dict:
    """
    Extract all activation values from multiple layers of the ESM model.

    Takes a batch of tokens, processes them through the model, and returns
    the representations from specified layers. Excludes padding tokens from
    the output. Note, this returns all all tokens flattened together so this
    is only really useful for training, not if you want to keep track of the
    activations for each sequence.

    Args:
        model: ESM model instance
        batch_tokens: Tokenized sequences
        layers: List of layer numbers to extract

    Returns:
        Dictionary mapping layer numbers to their activation tensors
    """
    with torch.no_grad():
        output = model(
            batch_tokens, attention_mask=batch_mask, output_hidden_states=True
        )
        token_representations = {
            layer: output.hidden_states[layer] for layer in layers}

    # Create a mask for non-padding tokens (tokens 0,1,2 are cls/pad/eos respectively)
    mask = batch_tokens > 3 if model_name.startswith("gLM2") else batch_tokens > 2 # NOTE: CHANGE TO > 3 FOR GLM (ESM2 is first 3 tokens as padding/cls/etc, glm is first 4)
    return {layer: rep[mask] for layer, rep in token_representations.items()}


"""
1. utils.fasta --> sequences, metadata (done)
2. sequences --> concat "<+>" (done)
3. embed_list_of_prot_seqs(gLM_650m) (in progress)

"""

# this will be a combination of embed_list_of_multimodal_seqs() but with the per-layer activation saving code below

def embed_fasta_file_for_all_layers_glm(
    model_name: str,
    fasta_file: Path,
    output_dir: Path,
    layers: List[int],
    shard_num: int,
    toks_per_batch: int = 1024,
    truncation_seq_length: int = 1022,    
    batch_size: int = 16,
):
    """
    Process a FASTA file through a gLM model and save layer activations.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"tattabio/{model_name}", trust_remote_code=True)
    model = AutoModel.from_pretrained(f"tattabio/{model_name}", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    alphabet = tokenizer.get_vocab()

    metadata, sequences = read_fasta(fasta_file) # method that seyone is currently writing to grab all sequences from the fasta file, which is a shard created by process_shard_range
    print(f"Read {fasta_file} with {len(sequences):,} sequences")

    ## Concatenate sequences with "<+>" separator
    sequences = ["<+>" + sequence for sequence in sequences]
    sequences = [sequence[:truncation_seq_length] for sequence in sequences]

    print(len(sequences))

    total_tokens = 0
    all_activations = {layer: [] for layer in layers}
    encodings = tokenizer(sequences, return_tensors='pt', padding=True)
    
    # reshape tokens into batches of size (num batches, batch_size, seq_length)
    print(encodings)
    toks = encodings.input_ids
    attention_mask = encodings.attention_mask

    dummy_seqs_needed = (len(toks) // batch_size + 1) * batch_size - len(toks)
    
    # Create dummy sequences for padding
    dummy_toks = torch.full((dummy_seqs_needed, toks.shape[1]), alphabet['<pad>'], dtype=torch.long)
    dummy_attention_mask = torch.zeros((dummy_seqs_needed, attention_mask.shape[1]), dtype=torch.long)
    
    # Concatenate dummy sequences to input_ids and attention_mask
    toks = torch.cat([toks, dummy_toks], dim=0)
    attention_mask = torch.cat([attention_mask, dummy_attention_mask], dim=0)

    # Reshape into batches
    reshaped_toks = toks.view(-1, batch_size, toks.shape[1])
    reshaped_attention_mask = attention_mask.view(-1, batch_size, attention_mask.shape[1])

    assert reshaped_toks.shape[0] == len(sequences) // batch_size + 1 if len(sequences) % batch_size != 0 else len(sequences) // batch_size

    print(reshaped_attention_mask.shape, reshaped_toks.shape)

    for batch_toks, batch_mask in tqdm(zip(reshaped_toks, reshaped_attention_mask), desc="Processing batches"):
        print("anotha one")
        batch_toks = batch_toks.to(device)
        batch_mask = batch_mask.bool().to(device)
        activations = get_activations(model,
                                      model_name,
                                      batch_toks,
                                      batch_mask,
                                      layers=layers)
        for layer in layers:
            all_activations[layer].append(activations[layer])

        # Count total tokens processed
        total_tokens += activations[layers[0]].shape[0]

        torch.cuda.empty_cache()

    # Save activations and metadata for each layer in the proper directory structure
    for layer in layers:
        layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = layer_output_dir / "activations.pt"
        metadata_file = layer_output_dir / "metadata.json"

        # Concatenate all activations for this layer
        layer_activations = torch.cat(all_activations[layer])

        # Shuffle the activations
        shuffled_indices = torch.randperm(total_tokens)
        layer_activations = layer_activations[shuffled_indices]

        # Save the tensor
        torch.save(layer_activations, output_file)
        print(f"Saved activations for layer {layer}, shard {shard_num} to {output_file}")

        # Save metadata
        metadata = {
            "model": model_name,
            "total_tokens": total_tokens,
            "d_model": model.config.dim,
            "dtype": str(layer_activations.dtype),
            "layer": layer,
            "shard": shard_num,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


    

def embed_fasta_file_for_all_layers(
    esm_model_name: str,
    fasta_file: Path,
    output_dir: Path,
    layers: List[int],
    shard_num: int,
    corrupt_esm: bool = False,
    toks_per_batch: int = 1024,
    truncation_seq_length: int = 1022,
):
    """
    Process a FASTA file through an ESM model and save layer activations.

    Processes sequences in batches, extracts activations from specified layers,
    shuffles the results, and saves them along with metadata. Uses GPU if available
    and not explicitly disabled.

    Args:
        model: ESM model instance
        alphabet: ESM alphabet for tokenization
        esm_model_name: Name of the ESM model being used
        fasta_file: Path to input FASTA file
        output_dir: Directory to save outputs
        layers: List of layer numbers to extract
        shard_num: Current shard number being processed
        toks_per_batch: Maximum tokens per batch
        truncation_seq_length: Maximum sequence length before truncation

    Outputs:
        - Saves activation tensors as .pt files
        - Saves metadata as JSON files
        - Creates directory structure for outputs
    """
    model, batch_converter, alphabet = get_model_converter_alphabet(
        esm_model_name, corrupt_esm, truncation_seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_converter,
        batch_sampler=batches,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Read {fasta_file} with {len(dataset):,} sequences")

    total_tokens = 0
    all_activations = {layer: [] for layer in layers}

    for (_, _, toks) in tqdm(data_loader, desc="Processing batches"):
        activations = get_activations(model,
                                      toks.to(device),
                                      (toks != alphabet.padding_idx).to(device),
                                      layers=layers)
        for layer in layers:
            all_activations[layer].append(activations[layer])

        # Count total tokens processed
        total_tokens += activations[layers[0]].shape[0]

        torch.cuda.empty_cache()

    # Save activations and metadata for each layer in the proper directory structure
    for layer in layers:
        layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = layer_output_dir / "activations.pt"
        metadata_file = layer_output_dir / "metadata.json"

        # Concatenate all activations for this layer
        layer_activations = torch.cat(all_activations[layer])

        # Shuffle the activations
        shuffled_indices = torch.randperm(total_tokens)
        layer_activations = layer_activations[shuffled_indices]

        # Save the tensor
        torch.save(layer_activations, output_file)
        print(f"Saved activations for layer {layer}, shard {shard_num} to {output_file}")

        # Save metadata
        metadata = {
            "model": esm_model_name,
            "total_tokens": total_tokens,
            "d_model": model.config.hidden_size,
            "dtype": str(layer_activations.dtype),
            "layer": layer,
            "shard": shard_num,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


def process_shard_range(
    fasta_dir: Path,
    output_dir: Path = Path("../../data/embeddings"),
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    start_shard: int | None = None,
    end_shard: int | None = None,
    corrupt_esm: bool = False,
    batch_size: int = 16,
):
    """
    Process a range of FASTA shards through an ESM model.

    Processes each shard in the specified range, extracting and saving activations
    from specified layers. Can optionally use a corrupted model with shuffled
    parameters. Skips shards that have already been processed.

    Args:
        start_shard: First shard number to process
        end_shard: Last shard number to process (inclusive)
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        fasta_dir: Directory containing FASTA shard files
        output_dir: Directory to save outputs
        corrupt_esm: Whether to shuffle model parameters

    Outputs:
        Creates directory structure with:
        - Activation tensors for each layer
        - Metadata files for each processed shard
    """

    # identify the number of shards in the fasta_dir
    fasta_files = list(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise ValueError(f"No FASTA files found in {fasta_dir}")

    if start_shard is None:
        start_shard = 0
    if end_shard is None:
        end_shard = len(fasta_files) - 1

    for i in range(start_shard, end_shard + 1):
        if esm_model_name[:3] == "esm":
            embed_fasta_file_for_all_layers(
                esm_model_name=esm_model_name,
                corrupt_esm=corrupt_esm,
                fasta_file=fasta_dir / f"shard_{i}.fasta",
                output_dir=output_dir,
                layers=layers,
                shard_num=i,
            )
        elif esm_model_name[:4] == "gLM2":
            #TODO: call GLM version
            embed_fasta_file_for_all_layers_glm(
                model_name=esm_model_name,
                fasta_file=fasta_dir / f"shard_{i}.fasta",
                output_dir=output_dir,
                layers=layers,
                shard_num=i,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Model not supported")

if __name__ == "__main__":
    from tap import tapify
    tapify(process_shard_range)
