## This is a fork of InterPLM that uses [gLM2](https://github.com/TattaBio/gLM2), a multimodal genomic language model, instead of a PLM. The commands in this README are not up to date but we are currently updating them.

# InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders
![Example feature activation patterns](https://github.com/user-attachments/assets/fc486fea-9303-45d3-aab2-9f9ceb51ac26)

InterPLM is a toolkit for extracting, analyzing, and visualizing interpretable features from protein language models (PLMs) using sparse autoencoders (SAEs). To learn more, check out [the preprint](https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1), or explore SAE features from every hidden layer of ESM-2-8M in our interactive dashboard, [InterPLM.ai](https://interPLM.ai).

### Key Features
- 🧬 Extract SAE features from protein language models
- 📊 Analyze and interpret learned features
- 🎨 Visualize feature patterns and relationships

## Example walkthrough
**_This walks through training, analysis, and feature visualization for SAEs based on PLM embeddings. The code is primarily set up for ESM-2 embeddings, but can easily be adapted to embeddings from any PLM._**

### 0. Setup
#### Installation
```bash
# Clone the repository
git clone https://github.com/ElanaPearl/interPLM.git
cd interPLM

# Create and activate conda environment
conda env create -f env.yml
conda activate interplm

# Install package
pip install -e .
```
#### Environment setup
Set the `INTERPLM_DATA` environment variable to establish the base directory for all data paths in this walkthrough (any downloaded .fasta files and ESM-2 embeddings created). If you don't want to use an environment variable, just replace `INTERPLM_DATA` with your path of choice throughout the walkthrough.
```bash
# For zsh (replace with .bashrc or preferred shell)
echo 'export INTERPLM_DATA="$HOME/your/preferred/path"' >> ~/.zshrc
source ~/.zshrc
```

### 1. Extract PLM embeddings for training data

**Obtain Sequences**
   - Download protein sequences (FASTA format) from [UniProt](https://www.uniprot.org/help/downloads)
   - In the paper, we use a random subset of UniRef50, but this is large and slow to download so for this walkthrough we'll use Swiss-Prot, which we have found also works for training SAEs.

```bash
# Download sequences
wget -P $INTERPLM_DATA/uniprot/ https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz

# Select random subset and filter to proteins with length < 1022 for ESM-2 compatibility
# Adjust num_proteins to increase the number of proteins kept
python interplm/data_processing/subset_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/uniprot_sprot.fasta.gz \
    --output_file $INTERPLM_DATA/uniprot/subset.fasta \
    --num_proteins 5000

# Shard fasta into smaller files to get embedded / shuffled
python interplm/data_processing/shard_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/subset.fasta \
    --output_dir $INTERPLM_DATA/uniprot_shards/ \
    --proteins_per_shard 1000  # 1000/shard -> ~0.5GB/shard for ESM-2 (8M)
```

**Generate ESM Embeddings for training**
```bash
# Adjust to include the layers you plan to train on, for the walkthrough we'll just use 3
python interplm/esm/fasta_to_sae_dataset.py \
    --fasta_dir $INTERPLM_DATA/uniprot_shards/ \
    --output_dir $INTERPLM_DATA/esm_embds/ \
    --esm_model_name esm2_t6_8M_UR50D \
    --layers 3
```

### 2. Train Sparse Autoencoders
```bash
# For training on ESM layer 3 embeddings extracted above
python interplm/train/train_plm_sae.py \
    --plm_embd_dir $INTERPLM_DATA/esm_embds/layer_3/ \
    --save_dir models/walkthrough_model/
```

**[Optional] If you want to track reconstruction fidelity:**

To track fidelity (how much the final model loss gets hurt by replacing embeddings with SAE reconstructions), you need a list of protein sequences to evaluate on. Here we provide a list of 1024 random sequences from [CATH](https://www.cathdb.info) but can use any list you'd like.
```bash
wget -P $INTERPLM_DATA/CATH/ https://interplm.s3.us-west-1.amazonaws.com/data/CATH/random_1k_subset.csv

python interplm/train/train_plm_sae.py \
    --plm_embd_dir $INTERPLM_DATA/esm_embds/layer_3/ \
    --save_dir models/walkthrough_model/ \
    --eval_seq_path $INTERPLM_DATA/CATH/random_1k_subset.csv
```
> Tracking reconstruction fidelity can be a helpful metric when evaluating SAE quality, but if you are getting nnsight errors, you can just run the first command.
### 3. Analyze associations between feature activations and UniProtKB annotations

1. Download Swiss-Prot (or any UniProtKB) data and extract quantitative binary concept labels then create eval sets from this. For our analysis in the paper, we just download all of Swiss-Prot (500k proteins) but for speed/memory efficiency of the walktrough, we add some filters to create a small subset of 800 proteins although it should be noted that scanning over a small dataset can lead to concept-feature pairings that don't generalize as well, but is fine for a walkthrough. <details> <summary> Details on data subset & customization </summary> The command below downloads annotations for proteins from mice with 3D structures, high quality annotations, and < 400 amino acids/protein, which leaves us with 800 sequences and a ~200kb download. If you want to include more data, remove `+AND+%28proteins_with%3A1%29+AND+%28annotation_score%3A5%29+AND+%28model_organism%3A10090%29+AND+%28length%3A%5B1+TO+400%5D%29` from the end of this query.</details>

```
# Download subset of Swiss-Prot
wget -O "${INTERPLM_DATA}/uniprotkb/proteins.tsv.gz" "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Clength%2Csequence%2Cec%2Cft_act_site%2Cft_binding%2Ccc_cofactor%2Cft_disulfid%2Cft_carbohyd%2Cft_lipid%2Cft_mod_res%2Cft_signal%2Cft_transit%2Cft_helix%2Cft_turn%2Cft_strand%2Cft_coiled%2Ccc_domain%2Cft_compbias%2Cft_domain%2Cft_motif%2Cft_region%2Cft_zn_fing%2Cxref_alphafolddb&format=tsv&query=%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29+AND+%28annotation_score%3A5%29+AND+%28model_organism%3A10090%29+AND+%28length%3A%5B1+TO+400%5D%29"

# Convert data into binary tabular annotations
python interplm/concept/extract_uniprotkb_annotations.py \
    --input_uniprot_path $INTERPLM_DATA/uniprotkb/proteins.tsv.gz \
    --output_dir $INTERPLM_DATA/uniprotkb/annotations \
    --n_shards 8 \
    --min_required_instances 10
```
>If using a larger subset of proteins, you will want to increase the number of shards the data is split into and also the minimum # of required instances for a concept to be included in analysis.

2. Convert the protein sequences to ESM embeddings
```
python interplm/esm/embed_uniprotkb.py \
    --input_dir $INTERPLM_DATA/uniprotkb/annotations/ \
    --output_dir $INTERPLM_DATA/uniprotkb/embeddings/ \
    --layer 3
```

3. Normalize the SAEs based on the max activating example across a random sample. UniRef50 or any other dataset can be used here for normalization, but we'll default to using the Swiss-Prot data we just embedded.
```
python interplm/sae/normalize.py \
    --sae_dir models/walkthrough_model/ \
    --esm_embds_dir $INTERPLM_DATA/uniprotkb/embeddings
```
4.  Create evaluation sets with different shards of data. Adjust the numbers here based on the number of shards created in Step 1. This step also filters out any concepts that have do not have many examples in your validation sets.
```
# Create validation set
python interplm/concept/prepare_concept_eval_set.py \
    --shards_to_include 0 1 2 3 \
    --uniprot_dir $INTERPLM_DATA/uniprotkb/annotations \
    --eval_name valid


# Create a test set
python interplm/concept/prepare_concept_eval_set.py \
    --shards_to_include 4 5 6 7 \
    --uniprot_dir $INTERPLM_DATA/uniprotkb/annotations \
    --eval_name test

```

6. Compare all features to all concepts at each threshold

```
for EVAL_SET in valid test
do
    # First track classification metrics (tp,fp,etc) on each shard
    python interplm/concept/compare_activations_to_concepts.py \
            --sae_dir models/walkthrough_model/ \
            --esm_embds_dir $INTERPLM_DATA/uniprotkb/embeddings \
            --eval_set_dir $INTERPLM_DATA/uniprotkb/annotations/${EVAL_SET}/ \
            --output_dir results/${EVAL_SET}_counts/ && \

    # Then combine all shards to calculate F1 scores
    python interplm/concept/calculate_f1.py \
    --eval_res_dir results/${EVAL_SET}_counts \
    --eval_set_dir $INTERPLM_DATA/uniprotkb/annotations/${EVAL_SET}/
done

# Report metrics on test set based on pairs selected in valid set
python interplm/concept/report_final_metrics.py \
    --valid_path results/valid_counts/concept_f1_scores.csv \
    --test_path results/test_counts/concept_f1_scores.csv
```

### 4. InterPLM Dashboard
The dashboard runs off a cache of pre-analyzed metrics (see [InterPLM.ai] for examples). Not all of these metrics are available in the repo yet but you can set up your own dashboard to visualize activation levels and concept results of your SAEs.

First set up a dashboard cache and track some proteins that activate each feature at various activation levels
```bash
python interplm/feature_vis/start_dashboard_cache.py \
    --sae_dir models/walkthrough_model \
    --esm_embeds_dir $INTERPLM_DATA/uniprotkb/embeddings \
    --aa_metadata_dir $INTERPLM_DATA/uniprotkb/annotations \
    --n_shards 4 \
    --esm_model_name esm2_t6_8M_UR50D \
    --layer 3 \
    --concept_dir results/test_counts

# Launch dashboard
cd interplm/dashboard && streamlit run app.py
```

>Note: After launching, access the dashboard at http://localhost:8501

If you've followed these steps, you'll have a dashboard for ESM2-8M layer 3 with 2,560 features, of which only 3 have concept associations - all related to coiled coils. On its own, this is not a particularly exciting set of features, but now you should be able to scale up both the training and concept-evaluation pipelines to explore a broader range of protein language model features. Increasing the training data, adjusting hyperparameters, and expanding the concept evaluation set will help identify features corresponding to other structural motifs, binding sites, and functional domains.

## Citation

If you use InterPLM in your research, please cite:

```bibtex
@article{simon2024interplm,
  title={InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders},
  author={Simon, Elana and Zou, James},
  journal={bioRxiv},
  pages={2024.11.14.623630},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

- Open an [issue](https://github.com/ElanaPearl/InterPLM/issues) on GitHub
- Email: epsimon [at] stanford [dot] edu
