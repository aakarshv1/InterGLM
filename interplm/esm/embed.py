""" Functions for embedding protein sequences using ESM models """

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from esm import FastaBatchedDataset, pretrained
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, EsmForMaskedLM

from interplm.utils import get_device


def get_model_converter_alphabet(esm_model_name: str, corrupt: bool = False, truncation_seq_length: int = 1022):
    """
    Initialize ESM model, batch converter, and alphabet for protein sequence processing.

    Args:
        esm_model_name: Name of the ESM model to load
        corrupt: If True, randomly shuffle model parameters. Defaults to False.
        truncation_seq_length: Maximum sequence length before truncation. Defaults to 1022.

    Returns:
        tuple: (model, batch_converter, alphabet)
            - model: Loaded ESM model
            - batch_converter: Function to convert sequences to model inputs
            - alphabet: ESM alphabet object for token conversion
    """
    device = get_device()
    _, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    model = EsmForMaskedLM.from_pretrained(
        f"facebook/{esm_model_name}").to(device)
    model.eval()

    if corrupt:
        model = shuffle_individual_parameters(model)

    batch_converter = alphabet.get_batch_converter(truncation_seq_length)

    return model, batch_converter, alphabet


def shuffle_individual_parameters(model, seed=42):
    """
    Randomly shuffle all parameters within a model while preserving their shapes.
    Used for creating controlled corrupted model baselines.

    Args:
        model: PyTorch model to shuffle
        seed: Random seed for reproducibility.

    Returns:
        Model with randomly shuffled parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for param in model.parameters():
        original_shape = param.data.shape
        flat_param = param.data.view(-1)
        shuffled_indices = torch.randperm(flat_param.nelement())
        shuffled_param = flat_param[shuffled_indices]
        param.data = shuffled_param.view(original_shape)

    return model

def embed_list_of_multimodal_seqs(
    seq_list: List[str],
    model_name: str,
    layer: int,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = None,
    device: torch.device = None,
):
    # NOTE: manually set, should be a parameter
    batch_size = 16
    if device is None:
        device = get_device()

    # Load model and tokenizer
    if model_name == "gLM2_150M":
        tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_150M", trust_remote_code=True)
        model = AutoModel.from_pretrained("tattabio/gLM2_150M", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    elif model_name == "gLM2_650M_embed":
        # same as above, but use path "tattabio/gLM2_650M_embed"
        tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M_embed", trust_remote_code=True)
        model = AutoModel.from_pretrained("tattabio/gLM2_650M_embed", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda
    else:
        raise ValueError(f"Model {model_name} not supported")
    model = model.to(device)
    model.eval()

    alphabet = tokenizer.get_vocab()
    ## Concatenate sequences with "<+>" separator
    seq_lst = ["<+>" + sequence for sequence in seq_list]
    seq_lst = [sequence[:truncation_seq_length] for sequence in seq_lst]

    print("Number of sequences: " + str(len(seq_lst)))

    encodings = tokenizer(seq_lst, return_tensors='pt', padding=True)
    
    # reshape tokens into batches of size (num batches, batch_size, seq_length)
    #print(encodings)
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

    assert reshaped_toks.shape[0] == len(seq_lst) // batch_size + 1 if len(seq_lst) % batch_size != 0 else len(seq_lst) // batch_size
    all_embeddings = [None] * len(seq_lst)  # Pre-allocate list

    batch_idx = 0 # batch index

    # NOTE: handle pre-forward pass attn_mask (created above) and post embedding (remove padding based on individual seqlen)
    for batch_toks, batch_mask in tqdm(zip(reshaped_toks, reshaped_attention_mask), desc="Processing batches"):
        # make sure last batch isn't full size if less than 16
        if len(seq_lst) - batch_idx * batch_size < batch_size:
            batch_toks = batch_toks[:len(seq_lst) - batch_idx * 16]
            batch_mask = batch_mask[:len(seq_lst) - batch_idx * 16]
        with torch.no_grad():
            embeddings = model(batch_toks.cuda(), attention_mask=batch_mask.cuda(), output_hidden_states=True)
            embeddings = embeddings.hidden_states[layer]  

        # Remove padding and special tokens, and store in the correct position
        for i, tokenized_seq in enumerate(batch_toks):
            #print(batch_toks.shape)
            seq_len = (tokenized_seq != alphabet['<pad>']).sum()
            # Extract original index based on current batch iter + seq index within batch
            seq_idx = batch_idx * batch_size + i
            #print(seq_idx)
            all_embeddings[seq_idx] = embeddings[i, :seq_len] # include '<+>' + sequence, exclude padding

        batch_idx += 1  # Increment batch index

        torch.cuda.empty_cache()

    # Verify that all sequences have been processed
    assert all(
        emb is not None for emb in all_embeddings), "Some sequences were not processed"

    return all_embeddings


def embed_list_of_prot_seqs(
    protein_seq_list: List[str],
    esm_model_name: str,
    layer: int,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = None,
    device: torch.device = None,
    corrupt: bool = False
) -> List[np.ndarray]:
    """
    Generate ESM embeddings for a list of protein sequences in batches.

    Args:
        protein_seq_list: List of protein sequences to embed
        esm_model_name: Name of the ESM model to use
        layer: Which transformer layer to extract embeddings from
        toks_per_batch Maximum tokens per batch. Defaults to 4096.
        truncation_seq_length: Maximum sequence length before truncation.
        device: Device to run computations on. Defaults to None.
        corrupt: If True, use corrupted model parameters. Defaults to False.

    Returns:
        List of embedding arrays, one per input sequence
    """
    if device is None:
        device = get_device()

    # Load ESM model
    model, batch_converter, alphabet = get_model_converter_alphabet(
        esm_model_name, corrupt, truncation_seq_length)

    # Create FastaBatchedDataset
    labels = [f"protein_{i}" for i in range(len(protein_seq_list))]
    dataset = FastaBatchedDataset(labels, protein_seq_list)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_converter,
        batch_sampler=batches,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Processing {len(dataset):,} sequences")
    total_tokens = 0
    all_embeddings = [None] * len(protein_seq_list)  # Pre-allocate list

    for labels, strs, toks in tqdm(data_loader, desc="Processing batches"):
        toks = toks.to(device)

        with torch.no_grad():
            results = model(toks, attention_mask=(
                toks != alphabet.padding_idx), output_hidden_states=True)
            embeddings = results.hidden_states[layer]

        # Remove padding and special tokens, and store in the correct position
        for i, (label, seq) in enumerate(zip(labels, strs)):
            seq_len = len(seq)
            # Extract original index from label
            seq_idx = int(label.split('_')[1])
            all_embeddings[seq_idx] = embeddings[i, 1:seq_len+1]

        total_tokens += embeddings.shape[0] * embeddings.shape[1]

    print(f"Processed {total_tokens:,} tokens in total")

    # Verify that all sequences have been processed
    assert all(
        emb is not None for emb in all_embeddings), "Some sequences were not processed"

    return all_embeddings



def embed_single_sequence(
    sequence: str,
    model_name: str,
    layer: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Embed a single protein or geomic sequence using various models.

    This method is optimized for quick, individual sequence processing, making it
    ideal for interactive applications like dashboards. Unlike batch processing
    methods, it doesn't use FastaBatchedDataset or complex data loading,
    making it more suitable for concurrent user queries.

    Args:
        sequence: Protein or genomic sequence string to embed.
        model_name: Name of the ESM model to use
        layer: Which transformer layer to extract embeddings from
        device: Computation device.

    Returns:
        Embedding tensor for the sequence, with shape
            (sequence_length, embedding_dimension)
    """
    if device is None:
        device = get_device()

    # Load model and tokenizer
    if model_name == "gLM2_150M":
        tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_150M", trust_remote_code=True)
        model = AutoModel.from_pretrained("tattabio/gLM2_150M", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    elif model_name == "gLM_650M_embed":
        # same as above, but use path "tattabio/gLM2_650M_embed"
        tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M_embed", trust_remote_code=True)
        model = AutoModel.from_pretrained("tattabio/gLM2_650M_embed", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
    model = model.to(device)
    model.eval()

    # Tokenize sequence
    if model_name == "gLM2_150M" or model_name == "gLM_650M_embed":
        inputs = tokenizer([sequence], return_tensors='pt')
        with torch.no_grad():
            embeddings = model(inputs.input_ids.to(device), output_hidden_states=True)
            embeddings = embeddings.hidden_states[layer]  
            embeddings = embeddings[0]
    else:
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get embeddings from specified layer
            embeddings = outputs.hidden_states[layer]
            # Remove batch dimension and special tokens - TODO: check if diff for GLM2
            embeddings = embeddings[0, 1:]

    return embeddings