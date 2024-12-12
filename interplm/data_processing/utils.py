import requests


def fetch_uniprot_sequence(uniprot_id: str) -> dict:
    """
    Fetch a protein sequence from UniProt's REST API and parse it.

    Args:
        uniprot_id (str): UniProt ID (e.g., 'Q1ACD6')

    Returns:
        dict: Dictionary containing 'uniprot_id', 'description', and 'sequence'
              or None if the sequence could not be fetched or parsed
    """
    # Construct the REST API URL
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    # Make the request
    response = requests.get(url)
    response.raise_for_status()  # Raises exception for 4XX/5XX status codes

    # Get the FASTA text
    fasta_text = response.text

    # Parse the FASTA format
    lines = fasta_text.strip().split('\n')
    if not lines or not lines[0].startswith('>'):
        return None

    header = lines[0]
    sequence = ''.join(lines[1:])

    # Parse header
    header_parts = header[1:].split('|')  # Remove '>' and split by '|'
    if len(header_parts) != 3:
        return None

    # Get description (everything after the last '|' up to OS=)
    full_desc = header_parts[2]
    description = full_desc.split(' OS=')[0]

    return {
        "Entry": uniprot_id,
        'Protein names': description,
        'Sequence': sequence
    }


def read_fasta(fasta_file):
    """Read a FASTA file and return a list of sequences and metadata"""
    sequences = []
    metadata = []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                metadata.append(line)
                sequences.append("")
            else:
                if "*" in line:
                    print("removing * from amino acid sequence. TODO, what are these?")
                    line = line.replace("*", "")
                sequences[-1] += line
    return metadata, sequences

