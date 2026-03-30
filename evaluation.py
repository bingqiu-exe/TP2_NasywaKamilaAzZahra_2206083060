import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(relevance_list, k=None):
    """ Discounted Cumulative Gain (DCG) dengan binary relevance. """
    if k is None:
        k = len(relevance_list)
    dcg_val = 0.0
    for i in range(min(k, len(relevance_list))):
        rel = relevance_list[i]
        if rel > 0:
            dcg_val += rel / math.log2(i + 2)   # i+2 karena rank dimulai dari 1 (log2(2)=1)
    return dcg_val


def ndcg(relevance_list, k=None):
    """ Normalized Discounted Cumulative Gain (NDCG). """
    actual = dcg(relevance_list, k)
    ideal_list = sorted(relevance_list, reverse=True)
    ideal = dcg(ideal_list, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def average_precision(relevance_list):
    """ Average Precision (AP) untuk binary relevance. """
    num_relevant = sum(relevance_list)
    if num_relevant == 0:
        return 0.0
    prec_sum = 0.0
    relevant_so_far = 0
    retrieved_so_far = 0
    for rel in relevance_list:
        retrieved_so_far += 1
        if rel == 1:
            relevant_so_far += 1
            prec_sum += relevant_so_far / retrieved_so_far
    return prec_sum / num_relevant

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []
    for qline in file:
        parts = qline.strip().split()
        qid = parts[0]
        query = " ".join(parts[1:])

        # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
        # yang tertera di qrels
        ranking = []
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
            did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
            ranking.append(qrels[qid][did])
        rbp_scores.append(rbp(ranking))
        dcg_scores.append(dcg(ranking))
        ndcg_scores.append(ndcg(ranking))
        ap_scores.append(average_precision(ranking))

    print("Hasil evaluasi TF-IDF terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score  =", sum(dcg_scores) / len(dcg_scores))
    print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
    print("AP score   =", sum(ap_scores) / len(ap_scores))

def eval_bm25(qrels, query_file="queries.txt", k=1000, k1=1.2, b=0.75):
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            ranking = []
            for score, doc in BSBI_instance.retrieve_bm25(query, k=k, k1=k1, b=b):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(average_precision(ranking))

    print("Hasil evaluasi BM25 terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score  =", sum(dcg_scores) / len(dcg_scores))
    print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
    print("AP score   =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)
  eval_bm25(qrels) 
