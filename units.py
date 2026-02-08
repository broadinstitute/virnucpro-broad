from Bio import SeqIO
from tqdm import tqdm
import os
import logging
import datetime

import torch
from transformers import AutoTokenizer, AutoModel
# fair-esm removed - extract_esm() deprecated, replaced by extract_fast_esm() in Phase 2

logger = logging.getLogger(__name__)

# Configuration toggle for dimension validation
VALIDATE_DIMS = os.getenv('VALIDATE_DIMS', 'true').lower() == 'true'

# Dimension constants - FastESM2_650 migration
DNA_DIM = 768       # DNABERT-S embedding dimension (unchanged)
PROTEIN_DIM = 1280  # FastESM2_650 embedding dimension (was 2560 for ESM2 3B)
MERGED_DIM = 2048   # DNA_DIM + PROTEIN_DIM (was 3328)

# Checkpoint versioning
CHECKPOINT_VERSION = "2.0.0"  # Breaking change from 1.x (ESM2 3B)


class DimensionError(Exception):
    """Raised when tensor dimensions don't match expected values."""

    def __init__(self, expected_dim, actual_dim, tensor_name=None, location=None):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.tensor_name = tensor_name or "unknown"
        self.location = location or "unknown"

        message = (
            f"Dimension mismatch at {self.location}: "
            f"Expected {self.expected_dim}-dim {self.tensor_name}, "
            f"got {self.actual_dim}-dim"
        )
        super().__init__(message)


def validate_protein_embeddings(proteins, data):
    """
    Validate protein embedding dimensions after extraction.
    Optional check (respects VALIDATE_DIMS setting).
    """
    if not VALIDATE_DIMS:
        return

    for protein, embedding in zip(proteins, data):
        if embedding.shape != (PROTEIN_DIM,):
            raise DimensionError(
                expected_dim=PROTEIN_DIM,
                actual_dim=embedding.shape[0] if embedding.dim() == 1 else embedding.shape[-1],
                tensor_name=f"protein_embedding[{protein}]",
                location="extract_fast_esm() output"
            )


def validate_merge_inputs(nucleotide_tensor, protein_tensor, seq_id):
    """
    Validate merge_data() input dimensions.
    Critical path - always runs regardless of VALIDATE_DIMS.
    """
    # DNA dimension check
    if nucleotide_tensor.shape != (DNA_DIM,):
        raise DimensionError(
            expected_dim=DNA_DIM,
            actual_dim=nucleotide_tensor.shape[0],
            tensor_name=f"dna_embedding[{seq_id}]",
            location="merge_data() input validation"
        )

    # Protein dimension check
    if protein_tensor.shape != (PROTEIN_DIM,):
        raise DimensionError(
            expected_dim=PROTEIN_DIM,
            actual_dim=protein_tensor.shape[0],
            tensor_name=f"protein_embedding[{seq_id}]",
            location="merge_data() input validation"
        )


def validate_merged_output(merged_tensor, seq_id):
    """
    Validate merge_data() output dimensions.
    Critical path - always runs.
    """
    if merged_tensor.shape != (MERGED_DIM,):
        raise DimensionError(
            expected_dim=MERGED_DIM,
            actual_dim=merged_tensor.shape[0],
            tensor_name=f"merged_features[{seq_id}]",
            location="merge_data() output"
        )

def split_fasta_chunk(input_file, output_file, chunk_size):
    with open(output_file, 'w') as out_handle:
        for record in SeqIO.parse(input_file, 'fasta'):
            sequence = record.seq
            seq_length = len(sequence)
            
            num_chunks = -(-seq_length // chunk_size)
            total_chunk_length = num_chunks * chunk_size
            repeat_length = total_chunk_length - seq_length
            repeat_region = repeat_length / num_chunks
            lower_int = int(repeat_region)
            upper_int = lower_int + 1
            
            low_up_numbers = [lower_int] * (num_chunks - 1)
            total_low_up_numbers = sum(low_up_numbers)
            need_to_add_up_nums = repeat_length - total_low_up_numbers
            final_low_up_numbers = [upper_int] * (need_to_add_up_nums) + [lower_int] * (num_chunks - 1 - (need_to_add_up_nums)) + [0]
            move_step = 0
            for a, b in zip(range(0, seq_length, chunk_size), final_low_up_numbers):
                if a > 1:
                    chunk = record[a-move_step:a-move_step + chunk_size]
                else:
                    chunk = record[a:a + chunk_size]
                new_record = chunk
                new_record.id = f"{record.id}_chunk_{a // chunk_size + 1}"
                new_record.description = ""
                SeqIO.write(new_record, out_handle, 'fasta')
                move_step += b

def create_refseq_pro_list(gbff_file):
    gbff = SeqIO.parse(gbff_file, "genbank")
    refseq_pro_dir = {}
    for record in gbff:
        protein_list = []
        for feature in record.features:
            if feature.type == "CDS":
                protein_seq = feature.qualifiers.get("translation", [""])[0]
                protein_list.append(protein_seq)
        refseq_pro_dir[record.id] = protein_list
    return refseq_pro_dir

def reverse_complement(sequence):
        complement = str.maketrans('ACGTacgt', 'TGCAtgca')
        return sequence.translate(complement)[::-1]

def translate_dna(sequence):
    codon_table = {
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*', 'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    }

    def translate_frame(sequence, frame):
        return ''.join(codon_table.get(sequence[i:i+3], '') for i in range(frame, len(sequence)-2, 3))
    
    frames = [translate_frame(sequence, frame) for frame in range(3)]
    rev_comp_sequence = reverse_complement(sequence)
    frames += [translate_frame(rev_comp_sequence, frame) for frame in range(3)]
    
    return frames

def seq_in_reflist(sequence, refseq_pro_list):
    if any(sequence in refpro for refpro in refseq_pro_list):
        return True
    else:
        return False

def identify_seq(seqid, sequence, refseq_pro_list = None, istraindata = False):
    final_list = []
    if istraindata:
        ambiguous_bases = {'N', 'R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D'}
        selected_seq_pro_dir = {'seqid': seqid, 'nucleotide': sequence, 'protein': ''}
        if any(base in sequence for base in ambiguous_bases):
            return None
        else:
            proteins_list = translate_dna(sequence)
            for num, item in enumerate(proteins_list, start=1):
                if istraindata:
                    if "*" not in item and seq_in_reflist(item, refseq_pro_list) == True:
                        pro_len = len(item)
                        seq_len = pro_len * 3
                        strand = "+" if num <= 3 else "-"
                        if strand == "+":
                            selected_seq = sequence[num-1:num+seq_len]
                            seqid_final = seqid + 'F' + str(num)
                        else:
                            selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                            seqid_final = seqid + 'R' + str(num - 3)
                        selected_seq_pro_dir = {
                            'seqid': seqid_final,
                            'nucleotide': selected_seq,
                            'protein': item
                        }
                        final_list.append(selected_seq_pro_dir)
                else:
                    if "*" not in item:
                        pro_len = len(item)
                        seq_len = pro_len * 3
                        strand = "+" if num <= 3 else "-"
                        if strand == "+":
                            selected_seq = sequence[num-1:num+seq_len]
                            seqid_final = seqid + 'F' + str(num)
                        else:
                            selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                            seqid_final = seqid + 'R' + str(num - 3)
                        selected_seq_pro_dir = {
                            'seqid': seqid_final,
                            'nucleotide': selected_seq,
                            'protein': item
                        }
                        final_list.append(selected_seq_pro_dir)
            return final_list
    else:
        proteins_list = translate_dna(sequence)
        selected_seq_pro_dir = {'seqid': seqid, 'nucleotide': sequence, 'protein': ''}
        for num, item in enumerate(proteins_list, start=1):
            if "*" not in item:
                pro_len = len(item)
                seq_len = pro_len * 3
                strand = "+" if num <= 3 else "-"
                if strand == "+":
                    selected_seq = sequence[num-1:num+seq_len]
                    seqid_final = seqid + 'F' + str(num)
                else:
                    selected_seq = reverse_complement(sequence)[num-3-1:num-3+seq_len]
                    seqid_final = seqid + 'R' + str(num - 3)
                selected_seq_pro_dir = {
                    'seqid': seqid_final,
                    'nucleotide': selected_seq,
                    'protein': item
                }
                final_list.append(selected_seq_pro_dir)
        return final_list



def process_record(index, record, refseq_pro_list = None, istraindata = False):
    sequence = str(record.seq).upper()
    seqid = record.id
    if istraindata:
        result = identify_seq(seqid, sequence, refseq_pro_list, istraindata)
    else:
        result = identify_seq(seqid, sequence)
    return (index, result)


def extract_DNABERT_S(input_file, out_file,
                      model_loaded = False, tokenizer = None, model = None):

    if model_loaded == False:
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    else:
        tokenizer = tokenizer
        model = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model.to(device)
    model.eval()

    data = []
    nucleotide = []

    records = list(SeqIO.parse(input_file, 'fasta'))
    for record in tqdm(records, desc="Processing sequences"):
        seq = str(record.seq)
        label = record.id

        inputs = tokenizer(seq, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            
            hidden_states = model(inputs["input_ids"])[0]
            embedding_mean = torch.mean(hidden_states, dim=1)

        result = {
            "label": label,
            "mean_representation": embedding_mean.squeeze().cpu().tolist()
        }
        nucleotide.append(label)
        data.append(result)

    if out_file is not None:
        torch.save({'nucleotide': nucleotide, 'data': data}, out_file)


def extract_esm(fasta_file,
                model_location='esm2_t36_3B_UR50D',
                truncation_seq_length=1024, toks_per_batch=2048,
                out_file=None, model_loaded = False, model = None, alphabet = None):
    raise NotImplementedError(
        "extract_esm() requires fair-esm which has been removed. "
        "Use extract_fast_esm() instead (implemented in Phase 2)."
    )

def get_batch_indices(sequence_lengths, toks_per_batch=2048):
    """
    Groups sequences into batches where total tokens (including BOS/EOS) does not exceed toks_per_batch.

    Args:
        sequence_lengths: list of sequence lengths
        toks_per_batch: maximum tokens per batch

    Returns:
        list of lists of indices (each inner list is one batch)
    """
    # Create list of (index, length) tuples and sort by length descending
    indexed_lengths = list(enumerate(sequence_lengths))
    indexed_lengths.sort(key=lambda x: x[1], reverse=True)

    batches = []
    current_batch = []
    current_tokens = 0

    for idx, seq_len in indexed_lengths:
        # Each sequence needs seq_len + 2 tokens (for BOS and EOS)
        tokens_needed = seq_len + 2

        # If single sequence exceeds limit, it gets its own batch
        if tokens_needed > toks_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([idx])
            continue

        # Check if adding this sequence would exceed the batch limit
        if current_tokens + tokens_needed > toks_per_batch:
            # Start new batch
            batches.append(current_batch)
            current_batch = [idx]
            current_tokens = tokens_needed
        else:
            # Add to current batch
            current_batch.append(idx)
            current_tokens += tokens_needed

    # Add final batch if not empty
    if current_batch:
        batches.append(current_batch)

    return batches

def validate_embeddings(proteins, data, expected_dim=None):
    """
    Validates embedding dimensions, NaN, and Inf values.

    Args:
        proteins: list of protein labels
        data: list of embedding tensors
        expected_dim: expected embedding dimension (defaults to PROTEIN_DIM)

    Returns:
        list of error strings (empty list if all valid)
    """
    if expected_dim is None:
        expected_dim = PROTEIN_DIM

    errors = []

    for protein, embedding in zip(proteins, data):
        # Check dimension
        if embedding.shape != (expected_dim,):
            errors.append(f"{protein}: Expected shape ({expected_dim},), got {embedding.shape}")

        # Check for NaN
        if torch.isnan(embedding).any():
            errors.append(f"{protein}: Contains NaN values")

        # Check for Inf
        if torch.isinf(embedding).any():
            errors.append(f"{protein}: Contains Inf values")

    return errors

def extract_fast_esm(fasta_file, out_file=None, model=None, tokenizer=None,
                     truncation_seq_length=1024, toks_per_batch=2048):
    """
    Extract FastESM2 embeddings from a FASTA file.

    Args:
        fasta_file: path to input FASTA file
        out_file: path to output .pt file (if None, don't save)
        model: pre-loaded FastESM2 model
        tokenizer: pre-loaded tokenizer
        truncation_seq_length: maximum sequence length before truncation
        toks_per_batch: maximum tokens per batch

    Returns:
        tuple of (proteins, data) where proteins is list of labels and data is list of 1D tensors
    """
    # Skip if output file already exists (resume capability)
    if out_file and os.path.exists(out_file):
        logger.info(f"Output file {out_file} already exists, loading and returning...")
        loaded = torch.load(out_file)
        return (loaded['proteins'], loaded['data'])

    logger.info(f"Extracting FastESM2 embeddings from {fasta_file}")

    # Read all sequences from FASTA
    records = list(SeqIO.parse(fasta_file, 'fasta'))
    sequences = [str(record.seq) for record in records]
    labels = [record.id for record in records]

    # Compute sequence lengths (capped at truncation_seq_length)
    sequence_lengths = [min(len(seq), truncation_seq_length) for seq in sequences]

    # Create batches using dynamic batching
    batch_indices = get_batch_indices(sequence_lengths, toks_per_batch=toks_per_batch)

    proteins = []
    data = []
    failures = []

    # Process batches with progress bar
    for batch_idx_list in tqdm(batch_indices, desc="Processing batches"):
        try:
            # Get sequences for this batch
            batch_seqs = [sequences[i] for i in batch_idx_list]
            batch_labels = [labels[i] for i in batch_idx_list]
            batch_seq_lengths = [sequence_lengths[i] for i in batch_idx_list]

            # Process batch
            with torch.no_grad():
                # Tokenize
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True,
                    max_length=truncation_seq_length + 2
                )

                # Move to device
                input_ids = inputs['input_ids'].to(model.device)
                attention_mask = inputs['attention_mask'].to(model.device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Mean pool excluding BOS/EOS for each sequence
                for i, (label, seq_len) in enumerate(zip(batch_labels, batch_seq_lengths)):
                    # Extract embeddings from positions 1 to seq_len+1 (excluding BOS at 0 and EOS after seq_len+1)
                    embedding = outputs.last_hidden_state[i, 1:seq_len+1].mean(0)

                    # Convert to float32 on CPU
                    embedding = embedding.float().cpu()

                    proteins.append(label)
                    data.append(embedding)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA OOM for batch of size {len(batch_idx_list)}, attempting to split batch")
            torch.cuda.empty_cache()

            # If batch has more than 1 sequence, split and retry
            if len(batch_idx_list) > 1:
                mid = len(batch_idx_list) // 2
                half1 = batch_idx_list[:mid]
                half2 = batch_idx_list[mid:]

                # Recursively process each half
                for half_batch in [half1, half2]:
                    try:
                        batch_seqs = [sequences[i] for i in half_batch]
                        batch_labels = [labels[i] for i in half_batch]
                        batch_seq_lengths = [sequence_lengths[i] for i in half_batch]

                        with torch.no_grad():
                            inputs = tokenizer(
                                batch_seqs,
                                return_tensors='pt',
                                padding='longest',
                                truncation=True,
                                max_length=truncation_seq_length + 2
                            )

                            input_ids = inputs['input_ids'].to(model.device)
                            attention_mask = inputs['attention_mask'].to(model.device)

                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                            for i, (label, seq_len) in enumerate(zip(batch_labels, batch_seq_lengths)):
                                embedding = outputs.last_hidden_state[i, 1:seq_len+1].mean(0)
                                embedding = embedding.float().cpu()
                                proteins.append(label)
                                data.append(embedding)

                    except Exception as sub_e:
                        error_msg = f"Failed to process half-batch: {str(sub_e)}"
                        logger.error(error_msg)
                        for label in batch_labels:
                            failures.append((label, 'processing_error', str(sub_e)))
            else:
                # Single sequence OOM - log and skip
                error_msg = f"Single sequence OOM: {batch_labels[0]}"
                logger.error(error_msg)
                failures.append((batch_labels[0], 'oom', str(e)))

        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            logger.error(error_msg)
            for idx in batch_idx_list:
                failures.append((labels[idx], 'batch_error', str(e)))

    # Log failures to extraction_failures.log
    if failures:
        with open('extraction_failures.log', 'a') as f:
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat()
            for seq_id, error_type, error_msg in failures:
                f.write(f"{timestamp} | {fasta_file} | {seq_id} | {error_type} | {error_msg}\n")
        logger.warning(f"Logged {len(failures)} failures to extraction_failures.log")

    # Validate embeddings
    validation_errors = validate_embeddings(proteins, data)
    if validation_errors:
        for error in validation_errors:
            logger.warning(f"Validation error: {error}")

        # If ALL embeddings failed, raise error
        if len(validation_errors) == len(proteins):
            raise RuntimeError(f"All embeddings failed validation for {fasta_file}")

    # Save output
    if out_file:
        torch.save({'proteins': proteins, 'data': data}, out_file)
        logger.info(f"Saved {len(proteins)} embeddings to {out_file}")

    logger.info(f"Extraction complete: {len(proteins)} proteins processed, {len(failures)} failures")

    return (proteins, data)

def split_fasta_file(input_file, output_dir, sequences_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_count = 0
    sequence_count = 0
    current_output_file = None

    with open(input_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                if sequence_count % sequences_per_file == 0:
                    if current_output_file:
                        current_output_file.close()
                    file_count += 1
                    current_output_file = open(os.path.join(output_dir, f'output_{file_count}.fa'), 'w')
                sequence_count += 1
            if current_output_file:
                current_output_file.write(line)

    if current_output_file:
        current_output_file.close()

def merge_data(DNABERT_S_data, ESM_data, merged_file, data_type = None):

    DNABERT_S_outfile = torch.load(DNABERT_S_data)
    ESM_outfile = torch.load(ESM_data)


    nucleotide_data_dict = {}
    protein_data_dict = {}

    merged_data = []

    for nucleotide, data in zip(DNABERT_S_outfile['nucleotide'], DNABERT_S_outfile['data']):
        nucleotide_data_dict[nucleotide] = torch.tensor(data['mean_representation'])
    for protein, data in zip(ESM_outfile['proteins'], ESM_outfile['data']):
        protein_data_dict[protein] = data

    for item in DNABERT_S_outfile['nucleotide']:
        if item in protein_data_dict and item in nucleotide_data_dict:
            protein_data = protein_data_dict[item]
            nucleotide_data = nucleotide_data_dict[item]

            # Critical path validation - always runs
            validate_merge_inputs(nucleotide_data, protein_data, item)

            merged_feature = torch.cat((nucleotide_data, protein_data), dim=-1)

            # Validate output
            validate_merged_output(merged_feature, item)

            merged_data.append(merged_feature)
        else:
            print(f"Warning: {item} not found in both datasets")

    merged_data = torch.stack(merged_data)
    if data_type == 'host':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [0]}
    elif data_type == 'viral':
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data, 'labels': [1]}
    elif data_type == None:
        merged_torch = {'ids': DNABERT_S_outfile['nucleotide'], 'data': merged_data}
    torch.save(merged_torch, merged_file)
