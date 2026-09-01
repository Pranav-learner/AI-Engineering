"""
Embedding Dimension Benchmark
==============================
AI Infrastructure Experiment: Vector Dimensionality Reduction & Search Quality

This script benchmarks the trade-offs between embedding dimensionality,
search latency, memory footprint, and retrieval quality (Recall@K).

Workflow:
1. Generate synthetic high-dimensional vector database and query vectors.
2. Compute ground-truth Top-K retrieval using exact Cosine Similarity.
3. Apply Principal Component Analysis (PCA) to reduce dimensions (1024D, 768D, 512D, 256D).
4. Measure search latency, Recall@K, and memory savings for each dimension.
5. Print a comparative benchmark summary.
"""

import time
import numpy as np
from sklearn.decomposition import PCA


# ==============================================================================
# 1. EXPERIMENT CONFIGURATION
# ==============================================================================
# Number of document vectors stored in our vector database
NUM_VECTORS = 10_000

# Number of search queries to execute during the benchmark
NUM_QUERIES = 100

# Original embedding dimension (e.g., standard OpenAI text-embedding-3-large is 1536D)
ORIGINAL_DIM = 1536

# Dimensions to evaluate via PCA dimensionality reduction
TARGET_DIMS = [1024, 768, 512, 256]

# Number of nearest neighbors to retrieve per query (Top-K)
TOP_K = 10

# Fixed seed for deterministic and reproducible benchmark runs
RANDOM_SEED = 42


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================
def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors to unit L2 norm (length = 1.0).
    
    Why is this important?
    When vectors have unit length, the dot product directly equals Cosine Similarity:
        CosineSimilarity(u, v) = (u . v) / (||u|| * ||v||) = u . v  (if ||u|| = ||v|| = 1)
    This allows us to compute cosine similarities for thousands of vectors using
    a single highly-optimized matrix multiplication (BLAS/GEMM).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Add a small epsilon to avoid division by zero in edge cases
    return vectors / (norms + 1e-10)


def compute_top_k(queries: np.ndarray, database: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """
    Computes exact Top-K nearest neighbors using cosine similarity and measures latency.
    
    Parameters:
        queries: (NUM_QUERIES, DIM) normalized query matrix
        database: (NUM_VECTORS, DIM) normalized database matrix
        k: number of nearest neighbors to retrieve
        
    Returns:
        top_k_indices: (NUM_QUERIES, k) indices of top-k vectors in database
        latency_ms_per_query: average search latency in milliseconds per query
    """
    start_time = time.perf_counter()
    
    # 1. Compute cosine similarity for all queries against all database vectors
    # Shape: (NUM_QUERIES, NUM_VECTORS)
    similarity_matrix = np.dot(queries, database.T)
    
    # 2. Find indices of the top K highest similarity scores for each query.
    # argpartition places the top-k elements at the end in O(N) time (faster than full argsort).
    partitioned_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
    
    # Sort only the top-k elements in descending order of similarity
    row_indices = np.arange(queries.shape[0])[:, None]
    top_k_scores = similarity_matrix[row_indices, partitioned_indices]
    sorted_order = np.argsort(-top_k_scores, axis=1)
    top_k_indices = np.take_along_axis(partitioned_indices, sorted_order, axis=1)
    
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query_ms = total_time_ms / queries.shape[0]
    
    return top_k_indices, latency_per_query_ms


def calculate_recall_at_k(ground_truth: np.ndarray, retrieved: np.ndarray) -> float:
    """
    Calculates Recall@K: The proportion of ground-truth relevant items that were retrieved.
    
    Recall@K = (Size of intersection between Ground Truth Top-K and Retrieved Top-K) / K
    Averaged across all benchmark queries.
    """
    total_recall = 0.0
    num_queries = ground_truth.shape[0]
    
    for i in range(num_queries):
        gt_set = set(ground_truth[i])
        ret_set = set(retrieved[i])
        intersection = gt_set.intersection(ret_set)
        total_recall += len(intersection) / float(len(gt_set))
        
    return total_recall / num_queries


def calculate_memory_mb(num_vectors: int, dim: int, bytes_per_element: int = 4) -> float:
    """
    Estimates raw memory footprint in Megabytes (MB) for storing vectors in float32.
    Float32 uses 4 bytes per dimension per vector.
    """
    total_bytes = num_vectors * dim * bytes_per_element
    return total_bytes / (1024 * 1024)


# ==============================================================================
# 3. BENCHMARK EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("AI INFRASTRUCTURE BENCHMARK: EMBEDDING DIMENSION REDUCTION (PCA)")
    print("=" * 80)
    print(f"• Dataset Size:     {NUM_VECTORS:,} vectors")
    print(f"• Query Batch Size: {NUM_QUERIES:,} queries")
    print(f"• Baseline Dim:     {ORIGINAL_DIM}D (float32)")
    print(f"• Target Dims:      {TARGET_DIMS}")
    print(f"• Metric:           Recall@{TOP_K} & Search Latency (ms/query)")
    print("=" * 80)
    print()

    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------------------------
    # Step 1: Generate Synthetic Database & Query Vectors
    # --------------------------------------------------------------------------
    print("[1/5] Generating synthetic high-dimensional vectors...")
    # Generate random Gaussian vectors to simulate embedding distributions
    raw_database = np.random.randn(NUM_VECTORS, ORIGINAL_DIM).astype(np.float32)
    raw_queries = np.random.randn(NUM_QUERIES, ORIGINAL_DIM).astype(np.float32)

    # Normalize vectors to unit length for cosine similarity
    norm_database_orig = normalize_vectors(raw_database)
    norm_queries_orig = normalize_vectors(raw_queries)
    print("      ✓ Vectors generated and L2-normalized successfully.")
    print()

    # --------------------------------------------------------------------------
    # Step 2: Compute Ground-Truth Top-K on Original High-Dimensional Vectors
    # --------------------------------------------------------------------------
    print(f"[2/5] Computing Ground-Truth Top-{TOP_K} on Original {ORIGINAL_DIM}D space...")
    ground_truth_indices, baseline_latency = compute_top_k(
        norm_queries_orig, norm_database_orig, TOP_K
    )
    baseline_mem = calculate_memory_mb(NUM_VECTORS, ORIGINAL_DIM)
    print(f"      ✓ Baseline Latency: {baseline_latency:.3f} ms/query | Memory: {baseline_mem:.2f} MB")
    print()

    # Store benchmark results for summary reporting
    results = [
        {
            "dim": f"{ORIGINAL_DIM} (Original)",
            "latency_ms": baseline_latency,
            "recall": 1.0,  # 100% by definition against itself
            "memory_mb": baseline_mem,
            "compression": "1.0x",
            "savings_pct": "0.0%",
        }
    ]

    # --------------------------------------------------------------------------
    # Step 3: Run PCA & Evaluate Each Reduced Dimension
    # --------------------------------------------------------------------------
    print("[3/5] Applying PCA & benchmarking reduced dimensions...")

    for target_dim in TARGET_DIMS:
        print(f"      -> Processing {target_dim}D...")
        pca_start = time.perf_counter()

        # Fit PCA on original database and transform database + queries
        pca = PCA(n_components=target_dim, random_state=RANDOM_SEED)
        db_reduced = pca.fit_transform(norm_database_orig)
        queries_reduced = pca.transform(norm_queries_orig)

        # L2-normalize the reduced vectors so dot product remains cosine similarity
        norm_db_reduced = normalize_vectors(db_reduced)
        norm_queries_reduced = normalize_vectors(queries_reduced)

        pca_time = (time.perf_counter() - pca_start) * 1000.0

        # Retrieve Top-K in reduced space
        reduced_indices, latency_ms = compute_top_k(
            norm_queries_reduced, norm_db_reduced, TOP_K
        )

        # Measure Recall@K against original space ground-truth
        recall_at_k = calculate_recall_at_k(ground_truth_indices, reduced_indices)

        # Calculate memory footprint and savings
        mem_mb = calculate_memory_mb(NUM_VECTORS, target_dim)
        compression_ratio = ORIGINAL_DIM / target_dim
        savings_pct = (1.0 - (target_dim / ORIGINAL_DIM)) * 100.0

        results.append({
            "dim": f"{target_dim}D",
            "latency_ms": latency_ms,
            "recall": recall_at_k,
            "memory_mb": mem_mb,
            "compression": f"{compression_ratio:.1f}x",
            "savings_pct": f"{savings_pct:.1f}%",
            "pca_fit_ms": pca_time
        })

    print("      ✓ All dimensions evaluated successfully.")
    print()

    # --------------------------------------------------------------------------
    # Step 4: Display Results Table
    # --------------------------------------------------------------------------
    print("[4/5] Benchmark Results:")
    print("-" * 85)
    header = (
        f"{'Dimension':<18} | {'Latency (ms/q)':<14} | {f'Recall@{TOP_K}':<10} | "
        f"{'Memory (MB)':<12} | {'Savings':<10} | {'Speedup':<8}"
    )
    print(header)
    print("-" * 85)

    for r in results:
        speedup = baseline_latency / r["latency_ms"] if r["latency_ms"] > 0 else 1.0
        row = (
            f"{r['dim']:<18} | "
            f"{r['latency_ms']:>12.3f} ms | "
            f"{r['recall'] * 100:>8.1f} % | "
            f"{r['memory_mb']:>10.2f} MB | "
            f"{r['savings_pct']:>8} | "
            f"{speedup:>6.2f}x"
        )
        print(row)
    print("-" * 85)
    print()

    # --------------------------------------------------------------------------
    # Step 5: Engineering Takeaways & Infrastructure Insights
    # --------------------------------------------------------------------------
    print("[5/5] Engineering Insights:")
    print("• Memory & Cost: Reducing from 1536D to 512D cuts RAM and vector index storage by 66.7%.")
    print("• Throughput: Lower dimensionality yields higher throughput and lower query latencies.")
    print("• Retrieval Quality Trade-off: Higher dimensions preserve fine-grained similarity rankings.")
    print("• Production Pattern: Often used in two-stage retrieval (e.g. coarse retrieval on 256D, rerank top-100 on 1536D).")
    print("=" * 80)


if __name__ == "__main__":
    main()
