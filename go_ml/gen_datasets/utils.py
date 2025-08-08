import requests
import sys
import argparse
from Bio import SeqIO
from io import StringIO

def fetch_sequence_from_redundant(accession_number):
    if(':' in accession_number):
        accession_number = accession_number.split(':')
    else:
        accession_number = [accession_number]
    for acc in accession_number:
        ret = fetch_sequence_from_uniprot(acc)
        if ret[0] is not None:
            return ret
    return None, None

def fetch_sequence_from_uniprot(accession_number):
    url = f"https://rest.uniprot.org/uniprotkb/{accession_number}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        fasta_content = response.text.strip()
        if not fasta_content:
            print(f"Warning: No content returned for {accession_number}.", file=sys.stderr)
            return None, None
        # Split the FASTA content into header and sequence
        lines = fasta_content.split('\n')
        if not lines:
            print(f"Warning: Empty FASTA content for {accession_number}.", file=sys.stderr)
            return None, None
        header = lines[0]
        # Join all lines after the header to form the complete sequence
        sequence = "".join(lines[1:])
        return header, sequence
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Error 404: Accession number '{accession_number}' not found in UniProtKB "
                  f"or it might be an identifier from another UniProt database (e.g., UniRef, UniParc) "
                  f"or an external database. This script is designed for UniProtKB accessions.", file=sys.stderr)
        else:
            print(f"HTTP Error fetching {accession_number}: {e}", file=sys.stderr)
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching {accession_number}: {e}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred for {accession_number}: {e}", file=sys.stderr)
        return None, None
    
import re
MAX_REQUESTS = 500  # Maximum number of requests to UniProt in a batch
def fetch_sequences_from_uniprot_batch(accession_numbers):
    print(len(accession_numbers))
    # Split the list into smaller chunks to avoid overwhelming the server
    sequences = []
    for i in range(0, len(accession_numbers), MAX_REQUESTS):
        batch_request = accession_numbers[i:i + MAX_REQUESTS]
        sequences.extend(fetch_sequences_from_uniprot_batch_helper(batch_request))
        print(f"Processed {len(sequences)} accessions, out of {len(accession_numbers)} total.")
    return sequences

def fetch_sequences_from_uniprot_batch_helper(accession_numbers):
    """
    Fetches amino acid sequences for a list of UniProt accession numbers
    using a single batch query to the UniProt REST API.

    Args:
        accession_numbers (list): A list of UniProt accession numbers (e.g., ['P0DTD1', 'Q9Y261']).

    Returns:
        list: A list of tuples, where each tuple is (header, sequence) for successfully
              retrieved entries. Returns an empty list if no sequences are retrieved.
    """
    if not accession_numbers:
        return []

    # Construct the query string for multiple accessions
    # Example: "accession:P0DTD1 OR accession:Q9Y261"
    query_parts = [f"accession:{acc}" for acc in accession_numbers]
    query_string = " OR ".join(query_parts)

    # UniProt REST API search endpoint for FASTA format
    # Using 'stream' endpoint for potentially large results, as suggested by UniProt docs for batch.
    # The 'size' parameter can be used with 'search' endpoint for smaller batches,
    # but 'stream' is generally better for large sets.
    # For simplicity and to ensure all results are returned, we use the 'stream' endpoint.
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query_string,
        "compressed": "false" # We want uncompressed text directly
    }

    try:
        print(f"Sending batch query for {len(accession_numbers)} accessions...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        fasta_content = response.text.strip()
        sequence_records = list(SeqIO.parse(StringIO(fasta_content), "fasta"))

        if not sequence_records:
            print("Warning: No content returned for the batch query. This might mean no sequences were found for the given accessions.", file=sys.stderr)
            return []
        retrieved_sequences = []
        for record in sequence_records:
            header = record.id.split('|')[1]  # Use the first part of the ID as the header
            sequence = str(record.seq)
            retrieved_sequences.append((header, sequence))

        retrieved_ids = set([header for header, _ in retrieved_sequences])
        failed_accessions = [acc for acc in accession_numbers if acc not in retrieved_ids]
        if failed_accessions:
            print(f"Warning: The following accession numbers were not found or could not be retrieved: {', '.join(failed_accessions)}", file=sys.stderr)
        retrieved_dict = {header: sequence for header, sequence in retrieved_sequences}
        ret = [(acc, retrieved_dict.get(acc, None)) for acc in accession_numbers]
        # print(len(accession_numbers), len(ret))
        return ret

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error during batch fetch: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        print(f"This might indicate issues with the query or that some accessions are not valid UniProtKB IDs.", file=sys.stderr)
        return []
    except requests.exceptions.RequestException as e:
        print(f"Network error during batch fetch: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred during batch fetch: {e}", file=sys.stderr)
        return []
    
# import requests as r
# from Bio import SeqIO
# from io import StringIO

# def uniprot2seq(ID: str):
#     baseUrl="http://www.uniprot.org/uniprot/"
#     currentUrl=baseUrl+ID+".fasta"
#     response = r.post(currentUrl)
#     cData=''.join(response.text)

#     Seq=StringIO(cData)
#     pSeq=list(SeqIO.parse(Seq,'fasta'))
#     try:
#         print(f"Sequence for {ID} found! Returning...{str(pSeq[0].seq)}")
#         return str(pSeq[0].seq)
#     except IndexError:
#         print(f"Sequence for {ID} not found! Returning...None")
#         return None

def parse_boundaries(boundaries):
    boundaries = boundaries.replace('[', '').replace(']', '')
    chunks = [b.strip() for b in boundaries.split(',')]
    ret = []
    for chunk in chunks:
        if '-' in chunk:
            start, end = chunk.split('-')
            ret.append((int(start.strip()), int(end.strip())))
        else:
            ret.append(int(chunk))
    return ret