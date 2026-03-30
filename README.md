# Search Engine Documentation

## Nasywa Kamila Az Zahra - 2206083060

## Overview

This project implements the Blocked Sort Based Indexing (BSBI)
The modules consist of:

- bsbi.py: contains an inverted index abstraction and logic for indexing using the Blocked Sort Based Indexing (BSBI) scheme.
- index.py: contains the basic logic for writing and reading indexes in bsbi.py.
- compression.py: contains the need to perform index compression.
- util.py: contains basic utilities for indexing.
- search.py: provides an example of how to perform a search process given several queries.
- evaluation.py: evaluate how effective your search engine is given a query relevance judgment (qrels.txt) and a fairly extensive list of queries (30 queries) in queries.txt.

## Features

**1. BSBI Indexing**
Splits the document collection into blocks (sub‑directories). For each block, it builds an in memory inverted index, writes it to disk as an intermediate index, and finally merges all intermediate indices into a single global inverted index using external merge sort.

**2. Variable‑Byte Encoding (VBE) Compression**
In a sorted postings list, docIDs can grow very large. By storing the difference (gap) between consecutive docIDs instead of the absolute values, we deal with much smaller integers, which are more compressible. Each byte in VBE uses 7 bits to store the actual numerical data. The 8th bit serves as a flag. In this implementation: a bit of 0 means more bytes follow for the current number and a bit of 1 means this is the last byte of the current number. In encoding process, it converts the postings_list into a list of gaps. Then, each gap is split into 7 bit chunks. Then, the last byte of each number is incremented by 128. All bytes are then joined into single bytes stream. In decoding process, reads one by one. If a byte is < 128, it shifts the current value and adds the bits. If it is >= 128, it treats it as the end of the number.

**3. Elias Gamma Compression**
Adds an alternative compression method to the existing Variable‑Byte Encoding (VBE), satisfying the assignment requirement for a bit‑level algorithm. In pre-processing, the posting list (list of document IDs) is transformed into a list of gaps: the first element stays as is then each subsequent element becomes the difference from the previous document ID. This produces many small integers (gaps), which Elias Gamma can compress efficiently. In encoding process, for a positive integer x: let n = floor(log2(x)). Write n zeros followed by a 1 (the unary prefix). Then append the binary representation of x without its leading 1.

**4. TF-IDF Retrieval**
Ranks documents using term frequency inverse document frequency. The query string is split into terms, and each term is converted to its unique term_id using the term_id_map. For every document containing the term, a weight is calculated as 1 + log(tf). This "dampens" the effect of high frequencies. It iterates through each query term, fetches the postings list, and adds the product of (document weight × IDF) to a score accumulator for each doc_id. The documents are sorted by score, and the top-k results are returned.

**5. BM25 Retrieval**
Implements the BM25 ranking function with parameters k1=1.2 and b=0.75. It uses document length normalisation and saturating term frequency. The system retrieves the average document length (avg_doc_len) from the index metadata. For IDF calculation, uses a specific BM25 IDF: log((N - df + 0.5) / (df + 0.5)). The b = 0.75 parameter ensures that if a document is much longer than the average.

**6. WAND for BM25**
It maintains per term upper bounds and skips large ranges of documents whose maximum possible score is below the current threshold.

## How to Run the Program

1. Open the terminal and run: python bsbi.py
   Can all sub‑directories in collection/, parse and index each document, write intermediate indices to the index/ folder, merge them into a final inverted index.
2. Run: python search.py
3. Run: python evaluation.py
