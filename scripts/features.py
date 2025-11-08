#!/usr/bin/env python3
"""
Complete Feature Extraction Pipeline for GNN Training
======================================================

This script runs the entire feature extraction pipeline:
1. Extract basic features (salesrank, reviews, ratings, categories)
2. Generate semantic embeddings using SBERT
3. Prepare graph data (nodes + edges) in JSON format

Usage:
    python run_pipeline.py                    # Full dataset
    python run_pipeline.py --sample 10000     # Test with 10K products
    python run_pipeline.py --batch-size 512   # Adjust batch size

Output:
    graph_data.json - Complete graph with nodes and edges for GNN training
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: EXTRACT BASIC FEATURES
# ============================================================================

def parse_amazon_books(filepath):
    """Parse amazon-books.txt and extract product information."""
    
    print("  Parsing products from amazon-books.txt...")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    products = content.split('\nId:')
    product_data = []
    
    for product in tqdm(products, desc="  Progress"):
        if not product.strip():
            continue
        
        lines = product.split('\n')
        prod = {
            'asin': None,
            'title': None,
            'salesrank': None,
            'categories': [],
            'review_count': 0,
            'avg_rating': 0.0
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('ASIN:'):
                prod['asin'] = line.split('ASIN:')[1].strip()
            elif line.startswith('title:'):
                prod['title'] = line.split('title:')[1].strip()
            elif line.startswith('salesrank:'):
                try:
                    prod['salesrank'] = int(line.split('salesrank:')[1].strip())
                except:
                    prod['salesrank'] = None
            elif line.startswith('|Books'):
                prod['categories'].append(line)
            elif line.startswith('reviews:'):
                parts = line.split()
                for j, part in enumerate(parts):
                    if part == 'total:' and j + 1 < len(parts):
                        prod['review_count'] = int(parts[j + 1])
                    elif part == 'rating:' and j + 1 < len(parts):
                        try:
                            prod['avg_rating'] = float(parts[j + 1])
                        except:
                            prod['avg_rating'] = 0.0
        
        if prod['asin'] and prod['title']:
            product_data.append(prod)
    
    return product_data


def extract_basic_features(products):
    """Extract basic numerical features from products."""
    
    print("  Extracting basic features...")
    basic_features = []
    
    for prod in tqdm(products, desc="  Progress"):
        if prod['salesrank'] is not None and prod['salesrank'] > 0:
            log_salesrank = np.log1p(prod['salesrank'])
        else:
            log_salesrank = np.log1p(10_000_000)
        
        if prod['categories']:
            category_depths = [cat.count('|') for cat in prod['categories']]
            category_depth = max(category_depths)
        else:
            category_depth = 0
        
        log_review_count = np.log1p(prod['review_count'])
        avg_rating = prod['avg_rating']
        
        basic_features.append([
            log_salesrank,
            category_depth,
            log_review_count,
            avg_rating
        ])
    
    return np.array(basic_features, dtype=np.float32)


# ============================================================================
# STEP 2: GENERATE SEMANTIC EMBEDDINGS
# ============================================================================

def prepare_text_for_embedding(products):
    """Prepare text for SBERT by combining title and category information."""
    
    print("  Preparing text for embeddings...")
    texts = []
    
    for prod in tqdm(products, desc="  Progress"):
        text_parts = [f"Title: {prod['title']}"]
        
        if prod['categories']:
            category_texts = []
            for cat_path in prod['categories']:
                clean_path = []
                parts = cat_path.split('|')
                for part in parts:
                    if '[' in part:
                        name = part.split('[')[0].strip()
                        if name and name not in ['Books',]:
                            clean_path.append(name)
                
                if clean_path:
                    category_texts.append(' → '.join(clean_path))
            
            if category_texts:
                text_parts.append(f"Categories: {', '.join(category_texts)}")
        
        texts.append(' '.join(text_parts))
    
    return texts


def generate_embeddings_batched(model, texts, batch_size=256):
    """Generate embeddings in batches."""
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Progress"):
        batch = texts[i:i + batch_size]
        batch_emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)


def generate_semantic_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=512, device=None):
    """Generate SBERT embeddings for texts."""
    
    print("  Loading SBERT model...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        print("  Error: sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        sys.exit(1)
    
    model = SentenceTransformer(model_name)
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    print(f"  Using device: {device}")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    print(f"  Generating embeddings for {len(texts):,} texts...")
    embeddings = generate_embeddings_batched(model, texts, batch_size)
    
    return embeddings


# ============================================================================
# STEP 3: PARSE EDGES
# ============================================================================

def parse_edges(data_file, valid_asins):
    """Parse co-purchase edges from amazon-books.txt."""
    
    print("  Parsing co-purchase edges...")
    
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    products = content.split('\nId:')
    edges = []
    
    for product in tqdm(products, desc="  Progress"):
        if not product.strip():
            continue
        
        lines = product.split('\n')
        source_asin = None
        
        for line in lines:
            if line.strip().startswith('ASIN:'):
                source_asin = line.split('ASIN:')[1].strip()
                break
        
        if not source_asin or source_asin not in valid_asins:
            continue
        
        for line in lines:
            line = line.strip()
            if line.startswith('similar:'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        num_similar = int(parts[1])
                        target_asins = parts[2:2+num_similar]
                        
                        for target_asin in target_asins:
                            if target_asin in valid_asins:
                                edges.append({
                                    'source': source_asin,
                                    'target': target_asin
                                })
                    except:
                        pass
                break
    
    return edges


# ============================================================================
# STEP 4: CREATE GRAPH DATA
# ============================================================================

def clean_category_path(cat_path):
    """Extract clean category names from a category path."""
    categories = []
    parts = cat_path.split('|')
    for part in parts:
        if '[' in part:
            name = part.split('[')[0].strip()
            if name and name not in ['Books']:
                categories.append(name)
    return categories


def create_node_dict(product, idx, sbert_embeddings, basic_features, max_category_depth):
    """Create a node dictionary in the required format."""
    
    # Extract basic features and convert to Python native types
    log_salesrank = float(basic_features[idx, 0])
    category_depth = int(basic_features[idx, 1])
    log_total_reviews = float(basic_features[idx, 2])
    avg_rating = float(basic_features[idx, 3])
    
    # Get semantic features
    semantic_features = [float(f) for f in sbert_embeddings[idx]]
    
    # Extract and clean categories
    categories = []
    if product['categories']:
        all_cats = set()
        for cat_path in product['categories']:
            cats = clean_category_path(cat_path)
            all_cats.update(cats)
        categories = sorted(list(all_cats))
    
    # Normalized category depth
    normalized_category_depth = round(float(category_depth) / float(max_category_depth), 2)
    
    # Convert salesrank to int
    salesrank = int(product['salesrank']) if product['salesrank'] else 0
    
    node = {
        'asin': str(product['asin']),
        'title': str(product['title']),
        'group': 'Book',
        'salesrank': salesrank,
        'log_salesrank': round(log_salesrank, 2),
        'avg_rating': round(avg_rating, 1),
        'total_reviews': int(product['review_count']),
        'log_total_reviews': round(log_total_reviews, 2),
        'categories': categories,
        'category_depth': category_depth,
        'normalized_category_depth': normalized_category_depth,
        'semantic_features': [round(f, 6) for f in semantic_features]
    }
    
    return node


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(input_file='data/amazon-books.txt', output_file='graph_data.json',
                batch_size=512, sample_size=None):
    """Run the complete feature extraction pipeline."""
    
    print("="*70)
    print("FEATURE EXTRACTION PIPELINE FOR GNN TRAINING")
    print("="*70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    if sample_size:
        print(f"Sample: {sample_size:,} products")
    print(f"Batch:  {batch_size}")
    
    # Step 1: Parse and extract basic features
    print("\n" + "-"*70)
    print("STEP 1/4: Extract Basic Features")
    print("-"*70)
    
    products = parse_amazon_books(input_file)
    print(f"  ✓ Parsed {len(products):,} products")
    
    # Optional sampling
    if sample_size and sample_size < len(products):
        print(f"  Sampling {sample_size:,} products...")
        indices = np.random.choice(len(products), sample_size, replace=False)
        products = [products[i] for i in indices]
        print(f"  ✓ Sampled {len(products):,} products")
    
    basic_features = extract_basic_features(products)
    print(f"  ✓ Basic features shape: {basic_features.shape}")
    
    # Step 2: Generate semantic embeddings
    print("\n" + "-"*70)
    print("STEP 2/4: Generate Semantic Embeddings")
    print("-"*70)
    
    texts = prepare_text_for_embedding(products)
    print(f"  ✓ Prepared {len(texts):,} texts")
    
    sbert_embeddings = generate_semantic_embeddings(texts, batch_size=batch_size)
    print(f"  ✓ Embeddings shape: {sbert_embeddings.shape}")
    
    # Step 3: Parse edges
    print("\n" + "-"*70)
    print("STEP 3/4: Parse Co-Purchase Edges")
    print("-"*70)
    
    valid_asins = {p['asin'] for p in products}
    edges = parse_edges(input_file, valid_asins)
    print(f"  ✓ Parsed {len(edges):,} edges")
    
    # Step 4: Create graph data
    print("\n" + "-"*70)
    print("STEP 4/4: Create Graph Data Structure")
    print("-"*70)
    
    print("  Creating node dictionaries...")
    nodes = []
    max_category_depth = basic_features[:, 1].max()
    
    for idx, product in enumerate(tqdm(products, desc="  Progress")):
        node = create_node_dict(product, idx, sbert_embeddings, basic_features, max_category_depth)
        nodes.append(node)
    
    print(f"  ✓ Created {len(nodes):,} nodes")
    
    # Create final structure
    graph_data = {
        'nodes': nodes,
        'edges': edges
    }
    
    # Save to JSON
    print(f"\n  Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    file_size = os.path.getsize(output_file) / (1024**2)
    print(f"  ✓ Saved! File size: {file_size:.2f} MB")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Avg degree: {2*len(edges)/len(nodes):.2f}")
    print(f"  Semantic features: {len(nodes[0]['semantic_features'])} dims")
    print(f"\n✅ Ready for GNN training!")
    
    # Print sample
    print("\n" + "-"*70)
    print("SAMPLE OUTPUT (first node, first 3 edges):")
    print("-"*70)
    sample = {
        'nodes': [nodes[0]],
        'edges': edges[:3]
    }
    print(json.dumps(sample, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Complete feature extraction pipeline for GNN training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                              # Full dataset
  python run_pipeline.py --sample 10000               # Test with 10K products
  python run_pipeline.py --batch-size 512             # Larger batch size
  python run_pipeline.py --output my_graph.json       # Custom output file
        """
    )
    
    parser.add_argument('--input', type=str, default='data/amazon-books.txt',
                       help='Input data file (default: data/amazon-books.txt)')
    parser.add_argument('--output', type=str, default='graph_data.json',
                       help='Output JSON file (default: graph_data.json)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for embeddings (default: 512)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (default: all products)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        sample_size=args.sample
    )


if __name__ == '__main__':
    main()
