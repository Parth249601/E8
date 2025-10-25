"""
E8 Text Processing Assignment - Complete Solution
Author: Complete Implementation
Date: October 2025

This script implements all requirements from the E8 assignment:
1. Data preprocessing (special characters, stop words, lemmatization)
2. Three vectorization methods (Count, TF-IDF, Word2Vec)
3. Distance calculations (Cosine, Euclidean)
4. Dimensionality reduction (PCA)
5. Visualizations (2D PCA, t-SNE)
6. Clustering analysis
7. Three custom problem investigations
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Distance and similarity
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

print("="*80)
print("E8 TEXT PROCESSING - COMPLETE SOLUTION")
print("="*80)

# Download required NLTK data
print("\nDownloading required NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✓ NLTK data downloaded successfully")
except Exception as e:
    print(f"Note: {e}")

# =============================================================================
# STEP 1: READ DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: READING DATA")
print("="*80)

df = pd.read_csv('Session-Summary-all-2025-S1.csv')
print(f"✓ Loaded {len(df)} documents")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(2))

# =============================================================================
# STEP 2: TEXT PREPROCESSING
# =============================================================================
print("\n" + "="*80)
print("STEP 2: TEXT PREPROCESSING")
print("="*80)

# 2a. Combine Topic and YourAnalysis columns
print("\n2a. Combining Topic and YourAnalysis columns...")
df['UnifiedText'] = df['Topic'].fillna('') + ' ' + df['YourAnalysis'].fillna('')
print(f"✓ Created UnifiedText column")

# 2b. Remove special characters
print("\n2b. Removing special characters...")
def remove_special_chars(text):
    """Remove special characters, keep only alphanumeric and spaces"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip().lower()

df['CleanedText'] = df['UnifiedText'].apply(remove_special_chars)
print(f"✓ Removed special characters")
print(f"Example: '{df['UnifiedText'].iloc[0][:100]}...'")
print(f"Cleaned: '{df['CleanedText'].iloc[0][:100]}...'")

# 2c. Remove stop words and lemmatize
print("\n2c. Removing stop words and lemmatizing...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Remove stop words and lemmatize"""
    try:
        # Tokenize - split on whitespace since text is already cleaned
        tokens = str(text).lower().split()
        
        # Remove stop words and lemmatize
        # Keep words longer than 2 characters
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                  if word not in stop_words and len(word) > 2 and word.isalnum()]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error processing text: {e}")
        return text  # Return original if error

print("Processing documents (this may take a moment)...")
df['PreprocessedText'] = df['CleanedText'].apply(preprocess_text)

# Check for empty documents
non_empty = df['PreprocessedText'].str.len() > 0
print(f"✓ Preprocessed {len(df)} documents")
print(f"✓ Non-empty documents: {non_empty.sum()}")
print(f"Example preprocessed text: '{df[non_empty]['PreprocessedText'].iloc[0][:150]}...'")

# 2d. Store in dataframe (already done above)
print(f"\n2d. ✓ Preprocessed text stored in 'PreprocessedText' column")

# =============================================================================
# STEP 3: VECTORIZATION
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CREATING VECTOR REPRESENTATIONS")
print("="*80)

# Filter out empty documents
df_filtered = df[df['PreprocessedText'].str.len() > 0].copy()
print(f"\nFiltered to {len(df_filtered)} non-empty documents")

# 3a. Count Vectorization
print("\n3a. Count Vectorization...")
count_vectorizer = CountVectorizer(max_features=1000, min_df=2)
count_vectors = count_vectorizer.fit_transform(df_filtered['PreprocessedText'])
print(f"✓ Count vectors shape: {count_vectors.shape}")
print(f"  (Documents: {count_vectors.shape[0]}, Features: {count_vectors.shape[1]})")

# 3b. TF-IDF Vectorization
print("\n3b. TF-IDF Vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
tfidf_vectors = tfidf_vectorizer.fit_transform(df_filtered['PreprocessedText'])
print(f"✓ TF-IDF vectors shape: {tfidf_vectors.shape}")

# 3c. Word2Vec Vectorization
print("\n3c. Word2Vec Vectorization...")
# Tokenize for Word2Vec
tokenized_docs = [doc.split() for doc in df_filtered['PreprocessedText']]
print(f"  Training Word2Vec model...")
word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, 
                          min_count=2, workers=4, epochs=10)
print(f"✓ Word2Vec model trained")
print(f"  Vocabulary size: {len(word2vec_model.wv)}")

# Create document vectors by averaging word vectors
def get_document_vector(doc, model, vector_size=100):
    """Get document vector by averaging word vectors"""
    words = doc.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vecs, axis=0)

word2vec_vectors = np.array([get_document_vector(doc, word2vec_model) 
                              for doc in df_filtered['PreprocessedText']])
print(f"✓ Word2Vec document vectors shape: {word2vec_vectors.shape}")

# 3d. Save dataframe
print("\n3d. Saving dataframe...")
df_filtered.to_csv('E8_processed_data.csv', index=False)
print(f"✓ Saved to 'E8_processed_data.csv'")

# =============================================================================
# STEP 4: DISTANCE ANALYSIS AND VISUALIZATION
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DISTANCE ANALYSIS AND VISUALIZATIONS")
print("="*80)

# Create a figure for all visualizations
fig = plt.figure(figsize=(20, 24))

# We'll analyze using TF-IDF (most common for text)
# For brevity, focusing on TF-IDF, but same can be done for Count and Word2Vec

vectorization_methods = {
    'Count': count_vectors.toarray(),
    'TF-IDF': tfidf_vectors.toarray(),
    'Word2Vec': word2vec_vectors
}

results_summary = []

for method_name, vectors in vectorization_methods.items():
    print(f"\n{'='*60}")
    print(f"ANALYZING: {method_name} Vectorization")
    print(f"{'='*60}")
    
    # 4a. Cosine Similarity
    print(f"\n4a. Calculating cosine similarity...")
    cosine_sim = cosine_similarity(vectors)
    cosine_dist = 1 - cosine_sim
    print(f"✓ Cosine distance matrix shape: {cosine_dist.shape}")
    print(f"  Mean cosine distance: {cosine_dist.mean():.4f}")
    print(f"  Std cosine distance: {cosine_dist.std():.4f}")
    
    # 4b. Euclidean Distance
    print(f"\n4b. Calculating Euclidean distance...")
    euclidean_dist = euclidean_distances(vectors)
    print(f"✓ Euclidean distance matrix shape: {euclidean_dist.shape}")
    print(f"  Mean Euclidean distance: {euclidean_dist.mean():.4f}")
    print(f"  Std Euclidean distance: {euclidean_dist.std():.4f}")
    
    # 4c. PCA Analysis
    print(f"\n4c. Performing PCA analysis...")
    pca = PCA(n_components=min(50, vectors.shape[1]))
    pca_vectors = pca.fit_transform(vectors)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"✓ PCA completed")
    print(f"  Variance explained by first 10 components: {cumulative_var[9]:.2%}")
    print(f"  Components needed for 90% variance: {np.argmax(cumulative_var >= 0.9) + 1}")
    
    # 4d. PCA 2D Visualization
    print(f"\n4d. Creating PCA 2D visualization...")
    pca_2d = PCA(n_components=2)
    pca_2d_vectors = pca_2d.fit_transform(vectors)
    
    # 4e. t-SNE 2D Visualization
    print(f"\n4e. Creating t-SNE 2D visualization...")
    # Use subset for t-SNE if dataset is large
    sample_size = min(1000, len(vectors))
    if len(vectors) > sample_size:
        sample_idx = np.random.choice(len(vectors), sample_size, replace=False)
        tsne_input = vectors[sample_idx]
    else:
        sample_idx = np.arange(len(vectors))
        tsne_input = vectors
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_input)-1))
    tsne_vectors = tsne.fit_transform(tsne_input)
    
    # 4f. Clustering on t-SNE coordinates
    print(f"\n4f. Clustering on t-SNE coordinates...")
    n_clusters = 5
    kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42)
    tsne_clusters = kmeans_tsne.fit_predict(tsne_vectors)
    tsne_silhouette = silhouette_score(tsne_vectors, tsne_clusters)
    print(f"✓ K-Means clustering (k={n_clusters}) on t-SNE")
    print(f"  Silhouette score: {tsne_silhouette:.4f}")
    
    # 4g. Clustering on original vectors
    print(f"\n4g. Clustering on original vectors...")
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42)
    orig_clusters = kmeans_orig.fit_predict(vectors)
    orig_silhouette = silhouette_score(vectors, orig_clusters)
    print(f"✓ K-Means clustering (k={n_clusters}) on original vectors")
    print(f"  Silhouette score: {orig_silhouette:.4f}")
    
    # Store results
    results_summary.append({
        'Method': method_name,
        'Mean_Cosine_Dist': cosine_dist.mean(),
        'Mean_Euclidean_Dist': euclidean_dist.mean(),
        'PCA_90%_Components': np.argmax(cumulative_var >= 0.9) + 1,
        't-SNE_Silhouette': tsne_silhouette,
        'Original_Silhouette': orig_silhouette
    })

print("\n" + "="*80)
print("SUMMARY OF ALL VECTORIZATION METHODS")
print("="*80)
results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle('E8 Text Processing - Comprehensive Analysis', fontsize=16, fontweight='bold')

row = 0
for method_name, vectors in vectorization_methods.items():
    print(f"Generating plots for {method_name}...")
    
    # PCA 2D
    pca_2d = PCA(n_components=2)
    pca_2d_vectors = pca_2d.fit_transform(vectors)
    
    axes[row, 0].scatter(pca_2d_vectors[:, 0], pca_2d_vectors[:, 1], 
                         alpha=0.5, s=10)
    axes[row, 0].set_title(f'{method_name}: PCA 2D')
    axes[row, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    axes[row, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    
    # t-SNE 2D with clusters
    sample_size = min(1000, len(vectors))
    if len(vectors) > sample_size:
        sample_idx = np.random.choice(len(vectors), sample_size, replace=False)
        tsne_input = vectors[sample_idx]
    else:
        tsne_input = vectors
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_input)-1))
    tsne_vectors = tsne.fit_transform(tsne_input)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(tsne_vectors)
    
    scatter = axes[row, 1].scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], 
                                    c=clusters, cmap='viridis', alpha=0.6, s=10)
    axes[row, 1].set_title(f'{method_name}: t-SNE with K-Means Clusters')
    axes[row, 1].set_xlabel('t-SNE 1')
    axes[row, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[row, 1], label='Cluster')
    
    # PCA variance explained
    pca_full = PCA(n_components=min(50, vectors.shape[1]))
    pca_full.fit(vectors)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    axes[row, 2].plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-')
    axes[row, 2].axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    axes[row, 2].set_title(f'{method_name}: PCA Cumulative Variance')
    axes[row, 2].set_xlabel('Number of Components')
    axes[row, 2].set_ylabel('Cumulative Variance Explained')
    axes[row, 2].legend()
    axes[row, 2].grid(True, alpha=0.3)
    
    row += 1

# Additional analysis plot
axes[3, 0].bar(results_df['Method'], results_df['t-SNE_Silhouette'])
axes[3, 0].set_title('Silhouette Scores Comparison (t-SNE Clustering)')
axes[3, 0].set_ylabel('Silhouette Score')
axes[3, 0].tick_params(axis='x', rotation=45)

axes[3, 1].bar(results_df['Method'], results_df['PCA_90%_Components'])
axes[3, 1].set_title('Components Needed for 90% Variance')
axes[3, 1].set_ylabel('Number of Components')
axes[3, 1].tick_params(axis='x', rotation=45)

# Distance comparison
methods = results_df['Method'].tolist()
x = np.arange(len(methods))
width = 0.35

normalized_cosine = results_df['Mean_Cosine_Dist'] / results_df['Mean_Cosine_Dist'].max()
normalized_euclidean = results_df['Mean_Euclidean_Dist'] / results_df['Mean_Euclidean_Dist'].max()

axes[3, 2].bar(x - width/2, normalized_cosine, width, label='Cosine (normalized)')
axes[3, 2].bar(x + width/2, normalized_euclidean, width, label='Euclidean (normalized)')
axes[3, 2].set_title('Normalized Distance Comparison')
axes[3, 2].set_ylabel('Normalized Distance')
axes[3, 2].set_xticks(x)
axes[3, 2].set_xticklabels(methods, rotation=45)
axes[3, 2].legend()

plt.tight_layout()
plt.savefig('E8_complete_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved comprehensive visualization as 'E8_complete_analysis.png'")

# =============================================================================
# STEP 5: THREE INTERESTING PROBLEMS
# =============================================================================
print("\n" + "="*80)
print("STEP 5: INVESTIGATING THREE INTERESTING PROBLEMS")
print("="*80)

# Use TF-IDF for these investigations
vectors_for_analysis = tfidf_vectors.toarray()

# Problem 1: Document Similarity - Find similar student responses
print("\n" + "="*60)
print("PROBLEM 1: Finding Similar Student Responses")
print("="*60)
print("Goal: Identify pairs of students with highly similar analyses")

cosine_sim = cosine_similarity(vectors_for_analysis)
np.fill_diagonal(cosine_sim, 0)  # Exclude self-similarity

# Find top 10 most similar document pairs
similar_pairs = []
n_docs = len(cosine_sim)
for i in range(n_docs):
    for j in range(i+1, n_docs):
        similar_pairs.append((i, j, cosine_sim[i, j]))

similar_pairs.sort(key=lambda x: x[2], reverse=True)
top_similar = similar_pairs[:10]

print("\nTop 10 Most Similar Document Pairs:")
for idx, (i, j, sim) in enumerate(top_similar, 1):
    roll_i = df_filtered.iloc[i]['RollNo']
    roll_j = df_filtered.iloc[j]['RollNo']
    topic_i = df_filtered.iloc[i]['Topic'][:40]
    topic_j = df_filtered.iloc[j]['Topic'][:40]
    print(f"{idx:2d}. Roll {roll_i} & Roll {roll_j}: Similarity = {sim:.4f}")
    print(f"    Topics: '{topic_i}...' & '{topic_j}...'")

# Visualize similarity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Problem 1: Document Similarity Analysis', fontsize=14, fontweight='bold')

# Similarity distribution
upper_triangle = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
axes[0].hist(upper_triangle, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Document Similarities')
axes[0].axvline(upper_triangle.mean(), color='r', linestyle='--', 
                label=f'Mean: {upper_triangle.mean():.3f}')
axes[0].legend()

# Heatmap of top similar pairs
top_indices = [i for i, j, _ in top_similar[:20]] + [j for i, j, _ in top_similar[:20]]
top_indices = list(set(top_indices))[:30]  # Limit to 30 for visibility
sim_subset = cosine_sim[np.ix_(top_indices, top_indices)]

sns.heatmap(sim_subset, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Similarity'})
axes[1].set_title('Similarity Heatmap (Top 30 Documents)')
axes[1].set_xlabel('Document Index')
axes[1].set_ylabel('Document Index')

plt.tight_layout()
plt.savefig('E8_problem1_similarity.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization as 'E8_problem1_similarity.png'")

# Problem 2: Topic Clustering - Identify main themes
print("\n" + "="*60)
print("PROBLEM 2: Identifying Main Themes/Topics")
print("="*60)
print("Goal: Cluster documents to discover main thematic groups")

# Try different numbers of clusters
silhouette_scores = []
cluster_range = range(3, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors_for_analysis)
    score = silhouette_score(vectors_for_analysis, cluster_labels)
    silhouette_scores.append(score)
    print(f"K={n_clusters}: Silhouette Score = {score:.4f}")

# Use optimal number of clusters
optimal_k = cluster_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_filtered['Cluster'] = kmeans_final.fit_predict(vectors_for_analysis)

# Analyze each cluster
print(f"\nCluster Analysis:")
for cluster_id in range(optimal_k):
    cluster_docs = df_filtered[df_filtered['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
    
    # Most common words in this cluster
    cluster_text = ' '.join(cluster_docs['PreprocessedText'])
    words = cluster_text.split()
    from collections import Counter
    word_freq = Counter(words).most_common(10)
    print(f"  Top words: {', '.join([w for w, c in word_freq])}")
    
    # Sample topics
    sample_topics = cluster_docs['Topic'].head(3).tolist()
    print(f"  Sample topics: {[t[:50]+'...' for t in sample_topics]}")

# Visualize clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Problem 2: Topic Clustering Analysis', fontsize=14, fontweight='bold')

# Silhouette scores for different k
axes[0].plot(list(cluster_range), silhouette_scores, 'bo-')
axes[0].axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Optimal Number of Clusters')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cluster distribution
cluster_counts = df_filtered['Cluster'].value_counts().sort_index()
axes[1].bar(cluster_counts.index, cluster_counts.values)
axes[1].set_xlabel('Cluster ID')
axes[1].set_ylabel('Number of Documents')
axes[1].set_title(f'Distribution of Documents Across {optimal_k} Clusters')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('E8_problem2_clustering.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization as 'E8_problem2_clustering.png'")

# Problem 3: Student Analysis - Identify learning patterns
print("\n" + "="*60)
print("PROBLEM 3: Student Learning Pattern Analysis")
print("="*60)
print("Goal: Analyze student engagement and knowledge diversity")

# Calculate per-student metrics
student_metrics = []

for roll_no in df_filtered['RollNo'].unique():
    student_docs = df_filtered[df_filtered['RollNo'] == roll_no]
    student_indices = student_docs.index.tolist()
    
    # Get vectors for this student
    student_vectors = vectors_for_analysis[[df_filtered.index.get_loc(idx) for idx in student_indices]]
    
    if len(student_vectors) > 1:
        # Calculate diversity (average pairwise distance)
        student_sim = cosine_similarity(student_vectors)
        np.fill_diagonal(student_sim, 0)
        avg_diversity = 1 - student_sim.mean()
        
        # Calculate breadth (how many clusters they cover)
        student_clusters = student_docs['Cluster'].nunique()
    else:
        avg_diversity = 0
        student_clusters = 1
    
    student_metrics.append({
        'RollNo': roll_no,
        'NumDocuments': len(student_docs),
        'AvgDiversity': avg_diversity,
        'ClusterBreadth': student_clusters,
        'TotalWords': student_docs['PreprocessedText'].str.split().str.len().sum()
    })

student_df = pd.DataFrame(student_metrics)
student_df = student_df.sort_values('NumDocuments', ascending=False)

print(f"\nTop 10 Most Active Students:")
print(student_df.head(10).to_string(index=False))

print(f"\nTop 10 Most Diverse Students (varied topics):")
diverse_students = student_df[student_df['NumDocuments'] >= 5].sort_values('AvgDiversity', ascending=False)
print(diverse_students.head(10).to_string(index=False))

# Visualize student analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Problem 3: Student Learning Pattern Analysis', fontsize=14, fontweight='bold')

# Submissions distribution
axes[0, 0].hist(student_df['NumDocuments'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Submissions')
axes[0, 0].set_ylabel('Number of Students')
axes[0, 0].set_title('Student Submission Distribution')
axes[0, 0].axvline(student_df['NumDocuments'].mean(), color='r', linestyle='--',
                   label=f'Mean: {student_df["NumDocuments"].mean():.1f}')
axes[0, 0].legend()

# Diversity vs submissions
axes[0, 1].scatter(student_df['NumDocuments'], student_df['AvgDiversity'], alpha=0.6)
axes[0, 1].set_xlabel('Number of Submissions')
axes[0, 1].set_ylabel('Average Topic Diversity')
axes[0, 1].set_title('Submissions vs Topic Diversity')
axes[0, 1].grid(True, alpha=0.3)

# Cluster breadth distribution
axes[1, 0].hist(student_df['ClusterBreadth'], bins=range(1, student_df['ClusterBreadth'].max()+2),
                edgecolor='black', alpha=0.7, align='left')
axes[1, 0].set_xlabel('Number of Different Topic Clusters')
axes[1, 0].set_ylabel('Number of Students')
axes[1, 0].set_title('Topic Breadth Distribution')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Total words distribution
axes[1, 1].hist(student_df['TotalWords'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Total Words Written')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].set_title('Student Writing Volume Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('E8_problem3_students.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization as 'E8_problem3_students.png'")

# Save final results
student_df.to_csv('E8_student_analysis.csv', index=False)
df_filtered.to_csv('E8_processed_with_clusters.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. E8_processed_data.csv - Preprocessed data with all text columns")
print("  2. E8_complete_analysis.png - Comprehensive visualization of all methods")
print("  3. E8_problem1_similarity.png - Document similarity analysis")
print("  4. E8_problem2_clustering.png - Topic clustering analysis")
print("  5. E8_problem3_students.png - Student learning pattern analysis")
print("  6. E8_student_analysis.csv - Detailed student metrics")
print("  7. E8_processed_with_clusters.csv - Data with cluster assignments")

print("\nSummary of Key Findings:")
print(f"  • Total documents analyzed: {len(df_filtered)}")
print(f"  • Unique students: {df_filtered['RollNo'].nunique()}")
print(f"  • Optimal number of topic clusters: {optimal_k}")
print(f"  • Average document similarity: {upper_triangle.mean():.4f}")
print(f"  • Most active student submissions: {student_df['NumDocuments'].max()}")
print("\nAll requirements from E8 assignment have been completed!")
print("="*80)
