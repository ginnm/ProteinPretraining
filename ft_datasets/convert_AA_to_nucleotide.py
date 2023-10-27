#!/bin/python3

import argparse

from typing import List
from copy import deepcopy

from Bio import SeqIO
from Bio.Seq import Seq

# Define fixed AA_Codon substitution
# From: https://www.ncbi.nlm.nih.gov/Taxonomy/taxonomyhome.html/index.cgi?chapter=tgencodes (transl_table=2; Verterbate Mitochondrial)

AA_to_codon = {
    "A":"GCT",
    "C":"TGT",
    "D":"GAT",
    "E":"GAA",
    "F":"TTT",
    "G":"GGT",
    "H":"CAT",
    "I":"ATT",
    "K":"AAA",
    "L":"TTA",
    "M":"ATA",
    "N":"AAT",
    "P":"CCA",
    "Q":"CAA",
    "R":"CGT",
    "S":"AGT",
    "T":"ACT",
    "V":"GTT",
    "W":"TGA",
    "Y":"TAT",
    "Z":"CAA",

    # Rare
    "O":"UAG",
    
    # Ambiguous
    "U":"AGA",
    "B":"AAC",
    "X":"NNN",

    # Start
    ">": "AUG",
    
     # Stop
    "*":"TAA",
}


def convert(seqrecord_path:str) -> List:

    nucleotide_seqrecords = list()

    for protein in SeqIO.parse(open(seqrecord_path, mode='r'), 'fasta'):
        seq = protein.seq
        nuc_seq = ""

        # Iteratively substitute sequence from beginning to end -- if you try to map non-iteratively,
        # you will end up with chimera sequences
        for char in seq:
            nuc_seq += AA_to_codon[char]

        nucleotide_protein = deepcopy(protein)
        nucleotide_protein.seq = Seq(nuc_seq)
        
        nucleotide_seqrecords.append(nucleotide_protein)
        
    return nucleotide_seqrecords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert AA FASTA to nucleotide FASTA.')
    parser.add_argument('FASTA_path', metavar='/path/to/FASTA.fasta', type=str, nargs=1,
                        help='The path to the AA FASTA file.')
    
    arguments = parser.parse_args()
    FASTA_path = arguments.FASTA_path[0]
    
    print(f"Starting conversion for file {FASTA_path}")
    
    nucleotide_FASTA_path = "".join(FASTA_path.split(".")[:-1]) + "_nucleotide.fasta"

    nucleotide_seqrecords = convert(arguments.FASTA_path[0])
    
    SeqIO.write(nucleotide_seqrecords, nucleotide_FASTA_path, "fasta")
    print(f"Conversion done, file stored at {nucleotide_FASTA_path}")
