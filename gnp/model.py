"""
Graph Neural Patching (GNP) Framework Implementation
Based on "Graph Neural Patching for Cold-Start Recommendations"

https://arxiv.org/pdf/2410.14241v1
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base directory: {BASE_DIR}")

DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"

FEATURES_PATH = DATA_DIR / "amazon_preprocessed_books.json"
TRAIN_SPLIT = SPLITS_DIR / "graph_data_train.json"
VAL_SPLIT = SPLITS_DIR / "graph_data_val.json"
TEST_SPLIT = SPLITS_DIR / "graph_data_test.json"

# TODO: Add output path

# NOTE: Works only for Macbook with M chip, change to cuda when on Windows, or remove entire line, but this can increase train time significantly.
# Set device for M3 Pro compatibility
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GNPDataset(Dataset):
    """
    Dataset class for handling GNP input data with separate feature and split files.
    Processes nodes with features from main file and edges from split files.
    """
    
    def __init__(self, 
                 features_path: str,
                 split_path: str,
                 negative_ratio: int = 4):
        """
        Initialize dataset from separate feature and split files.
        
        Args:
            features_path: Path to JSON file containing all nodes with features
            split_path: Path to JSON file containing split-specific nodes and edges
            negative_ratio: Number of negative samples per positive sample
        """
        # Load full feature data
        print(f"Loading features from {features_path}...")
        with open(features_path, 'r') as f:
            feature_data = json.load(f)
        
        # Load split data
        print(f"Loading split from {split_path}...")
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        # Create feature lookup dictionary
        self.feature_dict = {node['asin']: node for node in feature_data['nodes']}
        
        # Get nodes in this split
        self.split_node_asins = [node['asin'] for node in split_data['nodes']]
        self.split_edges = split_data['edges']
        
        # Create node index mapping (only for nodes in this split)
        self.node_to_idx = {asin: idx for idx, asin in enumerate(self.split_node_asins)}
        self.idx_to_node = {idx: asin for asin, idx in self.node_to_idx.items()}
        self.num_nodes = len(self.split_node_asins)
        
        print(f"Number of nodes in split: {self.num_nodes}")
        print(f"Number of nodes with features: {len(self.feature_dict)}")
        
        # Process node features
        self.node_features = self._process_node_features()
        
        # Process edges
        self.edge_index, self.edge_weights, self.edge_types = self._process_edges()
        
        # Determine warm vs cold nodes (nodes with edges are warm)
        self.warm_nodes = self._identify_warm_nodes()
        print(f"Warm nodes: {len(self.warm_nodes)}, Cold nodes: {self.num_nodes - len(self.warm_nodes)}")
        
        # Create positive and negative samples for training
        self.negative_ratio = negative_ratio
        self.positive_pairs = self._create_positive_pairs()
        self.negative_pairs = self._sample_negative_pairs()
        
        print(f"Positive pairs: {len(self.positive_pairs)}, Negative pairs: {len(self.negative_pairs)}")
    
    def _process_node_features(self) -> torch.Tensor:
        """
        Extract and concatenate node features:
        [log_salesrank, avg_rating, log_total_reviews, normalized_category_depth] + semantic_features
        Total dimension: 4 + 384 = 388
        """
        features = []
        
        for asin in self.split_node_asins:
            if asin in self.feature_dict:
                node = self.feature_dict[asin]
                basic_features = [
                    node['log_salesrank'],
                    node['avg_rating'],
                    node['log_total_reviews'],
                    node['normalized_category_depth']
                ]
                semantic_features = node['semantic_features']
                node_feat = basic_features + semantic_features
            else:
                # If node not in feature dict, use zero features
                print(f"Warning: Node {asin} not found in features, using zeros")
                node_feat = [0.0] * 388  # 4 basic + 384 semantic
            
            features.append(node_feat)
        
        return torch.tensor(features, dtype=torch.float32, device=device)
    
    def _process_edges(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Process edges into PyTorch Geometric format.
        Note: Edges in split files are already bidirectional.
        """
        edge_list = []
        weights = []
        edge_types = []
        
        for edge in self.split_edges:
            src_asin = edge['source']
            tgt_asin = edge['target']
            
            # Only include edges where both nodes are in this split
            if src_asin in self.node_to_idx and tgt_asin in self.node_to_idx:
                src_idx = self.node_to_idx[src_asin]
                tgt_idx = self.node_to_idx[tgt_asin]
                
                edge_list.append([src_idx, tgt_idx])
                weights.append(edge.get('weight', 1.0))
                edge_types.append(edge.get('type', 'real'))
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            edge_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_weights = torch.zeros(0, dtype=torch.float32, device=device)
        
        print(f"Processed {len(edge_list)} edges")
        return edge_index, edge_weights, edge_types
    
    def _identify_warm_nodes(self) -> set:
        """Identify warm nodes (nodes with at least one edge)."""
        warm_nodes = set()
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            tgt = self.edge_index[1, i].item()
            warm_nodes.add(src)
            warm_nodes.add(tgt)
        return warm_nodes
    
    def _create_positive_pairs(self) -> List[Tuple[int, int, float]]:
        """Create positive pairs from edges."""
        positive_pairs = []
        seen_pairs = set()
        
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            tgt = self.edge_index[1, i].item()
            weight = self.edge_weights[i].item()
            
            # Avoid duplicate pairs (since edges are bidirectional)
            pair_key = tuple(sorted([src, tgt]))
            if pair_key not in seen_pairs:
                positive_pairs.append((src, tgt, weight))
                seen_pairs.add(pair_key)
        
        return positive_pairs
    
    def _sample_negative_pairs(self) -> List[Tuple[int, int, float]]:
        """Sample negative node pairs for training."""
        negative_pairs = []
        positive_set = set()
        
        # Build set of all positive pairs (both directions)
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            tgt = self.edge_index[1, i].item()
            positive_set.add((src, tgt))
            positive_set.add((tgt, src))
        
        num_negatives = len(self.positive_pairs) * self.negative_ratio
        attempts = 0
        max_attempts = num_negatives * 100
        
        while len(negative_pairs) < num_negatives and attempts < max_attempts:
            u = random.randint(0, self.num_nodes - 1)
            i = random.randint(0, self.num_nodes - 1)
            
            if u != i and (u, i) not in positive_set:
                negative_pairs.append((u, i, 0.0))
            
            attempts += 1
        
        if len(negative_pairs) < num_negatives:
            print(f"Warning: Could only sample {len(negative_pairs)} negative pairs out of {num_negatives} requested")
        
        return negative_pairs
    
    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            u, i, label = self.positive_pairs[idx]
        else:
            u, i, label = self.negative_pairs[idx - len(self.positive_pairs)]
        
        # Ensure label is float32 for MPS compatibility
        return u, i, float(label)


class RandomWalkSampler:
    """
    Performs random walks on the graph for GWarmer.
    Pre-computes walks for efficiency as specified in the paper.
    """
    
    def __init__(self, edge_index: torch.Tensor, num_walks: int = 25, walk_length: int = 3):
        """
        Initialize random walk sampler.
        
        Args:
            edge_index: Edge indices of the graph
            num_walks: Number of random walks per node (S=25 from paper)
            walk_length: Length of each walk (K=3 from paper)
        """
        self.edge_index = edge_index
        self.num_walks = num_walks
        self.walk_length = walk_length
        
        # Build adjacency list for efficient sampling
        self.adj_list = self._build_adjacency_list()
    
    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        """Build adjacency list from edge index."""
        adj_list = defaultdict(list)
        if self.edge_index.shape[1] > 0:
            for src, dst in self.edge_index.t().cpu().numpy():
                adj_list[int(src)].append(int(dst))
        return dict(adj_list)
    
    def sample_walks(self, start_nodes: torch.Tensor) -> Dict[int, List[List[int]]]:
        """
        Sample random walks from given start nodes.
        
        Args:
            start_nodes: Tensor of node indices to start walks from
            
        Returns:
            Dictionary mapping node index to list of walks
        """
        walks = {}
        
        for node_idx in start_nodes.cpu().numpy():
            node_idx = int(node_idx)
            node_walks = []
            
            for _ in range(self.num_walks):
                walk = [node_idx]
                current = node_idx
                
                for _ in range(self.walk_length):
                    if current in self.adj_list and len(self.adj_list[current]) > 0:
                        next_node = random.choice(self.adj_list[current])
                        walk.append(next_node)
                        current = next_node
                    else:
                        # If no neighbors, stay at current node
                        walk.append(current)
                
                node_walks.append(walk)
            
            walks[node_idx] = node_walks
        
        return walks


class GWarmer(nn.Module):
    """
    GWarmer module for warm node recommendations.
    Implements random walk based subgraph aggregation with self-adaptive weighting.
    """
    
    def __init__(self, embedding_dim: int = 200, num_layers: int = 3, num_walks: int = 25):
        """
        Initialize GWarmer.
        
        Args:
            embedding_dim: Dimension of node embeddings
            num_layers: Number of subgraph layers (K=3)
            num_walks: Number of random walks per node (S=25)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_walks = num_walks
        
        # Self-adaptive weights for each layer (Eq. 4 in paper)
        self.user_weights = nn.Parameter(torch.ones(num_layers + 1))
        self.item_weights = nn.Parameter(torch.ones(num_layers + 1))
    
    def pool_walks(self, embeddings: torch.Tensor, walks: Dict[int, List[List[int]]], 
                   node_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform walk pooling step (Eq. 3 in paper).
        
        Args:
            embeddings: Node embedding matrix [num_nodes, embedding_dim]
            walks: Dictionary of random walks for each node
            node_indices: Indices of nodes to compute representations for
            
        Returns:
            Layer-wise representations [num_nodes, num_layers+1, embedding_dim]
        """
        batch_size = len(node_indices)
        layer_reprs = torch.zeros(batch_size, self.num_layers + 1, self.embedding_dim, device=device)
        
        for i, node_idx in enumerate(node_indices.cpu().numpy()):
            node_idx = int(node_idx)
            
            if node_idx not in walks or len(walks[node_idx]) == 0:
                # If no walks available, use node's own embedding for all layers
                layer_reprs[i, :, :] = embeddings[node_idx].unsqueeze(0).expand(self.num_layers + 1, -1)
                continue
            
            node_walks = walks[node_idx]
            
            # For each layer k, aggregate embeddings of nodes at distance k
            for k in range(self.num_layers + 1):
                if k == 0:
                    # Layer 0 is the node's own embedding
                    layer_reprs[i, k, :] = embeddings[node_idx]
                else:
                    # For layer k, collect nodes at position k in all walks
                    layer_nodes = []
                    for walk in node_walks:
                        if k < len(walk):
                            layer_nodes.append(walk[k])
                        else:
                            layer_nodes.append(walk[-1])  # Use last node if walk is shorter
                    
                    # Mean pooling (Eq. 3)
                    layer_node_embeddings = embeddings[layer_nodes]
                    layer_reprs[i, k, :] = layer_node_embeddings.mean(dim=0)
        
        return layer_reprs
    
    def forward(self, embeddings: torch.Tensor, walks: Dict[int, List[List[int]]], 
                node_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute GWarmer representations with self-adaptive weighting.
        
        Args:
            embeddings: Node embedding matrix
            walks: Dictionary of random walks
            node_indices: Indices of nodes to compute representations for
            
        Returns:
            GWarmer representations [num_nodes, embedding_dim]
        """
        # Get layer-wise representations
        layer_reprs = self.pool_walks(embeddings, walks, node_indices)
        
        # Determine if these are user or item nodes (use different weights)
        # For simplicity, we'll use user weights for the first half and item weights for second half
        # In a real bipartite graph, you'd have explicit user/item labels
        batch_size = len(node_indices)
        
        # Apply self-adaptive weighted sum (Eq. 4)
        weighted_reprs = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        for i in range(batch_size):
            # Use user weights for all nodes in this product graph
            weights = F.softmax(self.user_weights, dim=0)
            weighted_reprs[i] = torch.sum(layer_reprs[i] * weights.unsqueeze(1), dim=0)
        
        return weighted_reprs


class PatchingNetwork(nn.Module):
    """
    Patching Network for cold-start recommendations.
    Maps auxiliary features to embeddings with dropout mechanism.
    """
    
    def __init__(self, feature_dim: int = 388, embedding_dim: int = 200, dropout_ratio: float = 0.5):
        """
        Initialize Patching Network.
        
        Args:
            feature_dim: Dimension of auxiliary features
            embedding_dim: Dimension of output embeddings
            dropout_ratio: Dropout ratio for training (τ in paper)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout_ratio
        
        # Mapping functions f_U and f_I (Eq. 7)
        # Using a two-layer MLP
        self.user_mapper = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        self.item_mapper = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def dropout_gwarmer(self, gwarmer_repr: Optional[torch.Tensor], training: bool) -> torch.Tensor:
        """
        Apply dropout to GWarmer representations (Eq. 6 in paper).
        
        Args:
            gwarmer_repr: GWarmer representations or None for cold nodes
            training: Whether in training mode
            
        Returns:
            Dropped out representations
        """
        if gwarmer_repr is None or not training:
            # For cold nodes or during inference, use zero vector
            if gwarmer_repr is None:
                return torch.zeros(1, self.embedding_dim, device=device)
            else:
                return torch.zeros_like(gwarmer_repr)
        
        # During training, apply Bernoulli dropout (Eq. 6)
        dropout_mask = torch.bernoulli(torch.full_like(gwarmer_repr, self.dropout_ratio))
        dropped_repr = gwarmer_repr * (1 - dropout_mask)
        
        return dropped_repr
    
    def forward(self, features: torch.Tensor, gwarmer_repr: Optional[torch.Tensor], 
                training: bool, is_user: bool = True) -> torch.Tensor:
        """
        Map features to embeddings (Eq. 7-8 in paper).
        
        Args:
            features: Auxiliary features [batch_size, feature_dim]
            gwarmer_repr: GWarmer representations [batch_size, embedding_dim] or None
            training: Whether in training mode
            is_user: Whether these are user nodes (determines which mapper to use)
            
        Returns:
            Patching Network embeddings [batch_size, embedding_dim]
        """
        batch_size = features.shape[0]
        
        # Handle GWarmer representations
        if gwarmer_repr is None:
            dropped_repr = torch.zeros(batch_size, self.embedding_dim, device=device)
        else:
            dropped_repr = self.dropout_gwarmer(gwarmer_repr, training)
        
        # Concatenate dropped GWarmer repr with features (Eq. 7)
        combined = torch.cat([dropped_repr, features], dim=1)
        
        # Apply mapping function
        mapper = self.user_mapper if is_user else self.item_mapper
        embeddings = mapper(combined)
        
        return embeddings


class GNP(nn.Module):
    """
    Complete Graph Neural Patching (GNP) model.
    Combines GWarmer for warm recommendations and Patching Networks for cold-start.
    """
    
    def __init__(self, num_nodes: int, feature_dim: int = 388, embedding_dim: int = 200,
                 num_layers: int = 3, num_walks: int = 25, dropout_ratio: float = 0.5):
        """
        Initialize GNP model.
        
        Args:
            num_nodes: Total number of nodes in the graph
            feature_dim: Dimension of node features
            embedding_dim: Dimension of embeddings
            num_layers: Number of GWarmer layers (K)
            num_walks: Number of random walks (S)
            dropout_ratio: Dropout ratio for Patching Network (τ)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Initialize node embeddings (these serve as base embeddings E_u, E_i)
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # GWarmer module
        self.gwarmer = GWarmer(embedding_dim, num_layers, num_walks)
        
        # Patching Network
        self.patching_network = PatchingNetwork(feature_dim, embedding_dim, dropout_ratio)
    
    def compute_warm_scores(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                           user_walks: Dict[int, List[List[int]]], 
                           item_walks: Dict[int, List[List[int]]]) -> torch.Tensor:
        """
        Compute warm recommendation scores using GWarmer (Eq. 5).
        
        Args:
            user_indices: User node indices
            item_indices: Item node indices
            user_walks: Random walks for users
            item_walks: Random walks for items
            
        Returns:
            Warm scores (inner products)
        """
        base_embeddings = self.node_embeddings.weight
        
        # Get GWarmer representations
        user_repr = self.gwarmer(base_embeddings, user_walks, user_indices)
        item_repr = self.gwarmer(base_embeddings, item_walks, item_indices)
        
        # Compute inner product (Eq. 5)
        scores = torch.sum(user_repr * item_repr, dim=1)
        
        return scores
    
    def compute_cold_scores(self, user_features: torch.Tensor, item_features: torch.Tensor,
                           user_warm_emb: Optional[torch.Tensor], 
                           item_warm_emb: Optional[torch.Tensor],
                           training: bool) -> torch.Tensor:
        """
        Compute cold-start scores using Patching Network (Eq. 8).
        
        Args:
            user_features: User auxiliary features
            item_features: Item auxiliary features
            user_warm_emb: User GWarmer embeddings (None for cold users)
            item_warm_emb: Item GWarmer embeddings (None for cold items)
            training: Whether in training mode
            
        Returns:
            Cold-start scores (inner products)
        """
        # Get Patching Network embeddings (Eq. 7)
        user_emb = self.patching_network(user_features, user_warm_emb, training, is_user=True)
        item_emb = self.patching_network(item_features, item_warm_emb, training, is_user=False)
        
        # Compute inner product (Eq. 8)
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        return scores
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                user_features: torch.Tensor, item_features: torch.Tensor,
                warm_users: set, warm_items: set,
                user_walks: Dict[int, List[List[int]]], 
                item_walks: Dict[int, List[List[int]]],
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing Eq. 1 from paper.
        
        Args:
            user_indices: Batch of user indices
            item_indices: Batch of item indices
            user_features: User auxiliary features
            item_features: Item auxiliary features
            warm_users: Set of warm user indices
            warm_items: Set of warm item indices
            user_walks: Random walks for users
            item_walks: Random walks for items
            training: Whether in training mode
            
        Returns:
            Tuple of (warm_scores, cold_scores)
        """
        batch_size = len(user_indices)
        warm_scores = torch.zeros(batch_size, device=device)
        cold_scores = torch.zeros(batch_size, device=device)
        
        # Determine which pairs are warm (both nodes have edges)
        warm_mask = torch.tensor([
            (u.item() in warm_users and i.item() in warm_items)
            for u, i in zip(user_indices, item_indices)
        ], device=device)
        
        # Compute warm scores for warm pairs
        if warm_mask.any():
            warm_user_idx = user_indices[warm_mask]
            warm_item_idx = item_indices[warm_mask]
            
            warm_scores[warm_mask] = self.compute_warm_scores(
                warm_user_idx, warm_item_idx, user_walks, item_walks
            )
            
            # Also compute cold scores for warm pairs during training
            if training:
                base_embeddings = self.node_embeddings.weight
                
                # Get GWarmer representations for Patching Network training
                user_warm_emb = self.gwarmer(base_embeddings, user_walks, warm_user_idx)
                item_warm_emb = self.gwarmer(base_embeddings, item_walks, warm_item_idx)
                
                cold_scores[warm_mask] = self.compute_cold_scores(
                    user_features[warm_mask], item_features[warm_mask],
                    user_warm_emb, item_warm_emb, training=True
                )
        
        # Compute cold scores for cold pairs
        cold_mask = ~warm_mask
        if cold_mask.any():
            cold_scores[cold_mask] = self.compute_cold_scores(
                user_features[cold_mask], item_features[cold_mask],
                None, None, training=False
            )
        
        return warm_scores, cold_scores


def train_gnp(model: GNP, dataset: GNPDataset, num_epochs: int = 100, 
              batch_size: int = 256, learning_rate: float = 0.001) -> List[float]:
    """
    Train the GNP model.
    
    Args:
        model: GNP model instance
        dataset: GNPDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        
    Returns:
        List of training losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Pre-compute random walks for all nodes
    print("Pre-computing random walks...")
    walk_sampler = RandomWalkSampler(dataset.edge_index, num_walks=25, walk_length=3)
    all_nodes = torch.arange(dataset.num_nodes, device=device)
    all_walks = walk_sampler.sample_walks(all_nodes)
    print("Random walks computed!")
    
    losses = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            user_idx, item_idx, labels = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            # Get features for batch
            user_features = dataset.node_features[user_idx]
            item_features = dataset.node_features[item_idx]
            
            # Forward pass
            warm_scores, cold_scores = model(
                user_idx, item_idx,
                user_features, item_features,
                dataset.warm_nodes, dataset.warm_nodes,
                all_walks, all_walks,
                training=True
            )
            
            # Compute loss as specified in paper: MSE for both components (Eq. 9)
            warm_loss = F.mse_loss(warm_scores, labels)
            cold_loss = F.mse_loss(cold_scores, labels)
            total_loss = warm_loss + cold_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_gnp(model: GNP, dataset: GNPDataset, batch_size: int = 256) -> Dict[str, float]:
    """
    Evaluate the GNP model.
    
    Args:
        model: Trained GNP model
        dataset: GNPDataset instance
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Pre-compute random walks
    print("Pre-computing random walks for evaluation...")
    walk_sampler = RandomWalkSampler(dataset.edge_index, num_walks=25, walk_length=3)
    all_nodes = torch.arange(dataset.num_nodes, device=device)
    all_walks = walk_sampler.sample_walks(all_nodes)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            user_idx, item_idx, labels = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            
            # Get features
            user_features = dataset.node_features[user_idx]
            item_features = dataset.node_features[item_idx]
            
            # Forward pass
            warm_scores, cold_scores = model(
                user_idx, item_idx,
                user_features, item_features,
                dataset.warm_nodes, dataset.warm_nodes,
                all_walks, all_walks,
                training=False
            )
            
            # Use appropriate scores based on warm/cold status (Eq. 1)
            scores = torch.where(
                torch.tensor([
                    (u.item() in dataset.warm_nodes and i.item() in dataset.warm_nodes)
                    for u, i in zip(user_idx, item_idx)
                ], device=device),
                warm_scores,
                cold_scores
            )
            
            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Apply sigmoid for probability
    pred_probs = 1 / (1 + np.exp(-all_preds))
    pred_binary = (pred_probs > 0.5).astype(float)
    
    # Calculate accuracy
    accuracy = np.mean(pred_binary == all_labels)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred_probs - all_labels) ** 2))
    
    # Calculate AUC if we have both positive and negative samples
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, pred_probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'rmse': rmse,
        'auc': auc
    }


def main():
    """
    Main execution function for training and evaluating GNP model.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load datasets
    print("=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    # TODO: Load FEATURES_PATH once for performance
    train_dataset = GNPDataset(
        features_path=FEATURES_PATH,
        split_path=TRAIN_SPLIT,
        negative_ratio=4
    )
    
    val_dataset = GNPDataset(
        features_path=FEATURES_PATH,
        split_path=VAL_SPLIT,
        negative_ratio=4
    )
    
    test_dataset = GNPDataset(
        features_path=FEATURES_PATH,
        split_path=TEST_SPLIT,
        negative_ratio=4
    )
    
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print("=" * 80)
    print(f"Train - Nodes: {train_dataset.num_nodes}, Edges: {train_dataset.edge_index.shape[1]}, "
          f"Warm: {len(train_dataset.warm_nodes)}")
    print(f"Val   - Nodes: {val_dataset.num_nodes}, Edges: {val_dataset.edge_index.shape[1]}, "
          f"Warm: {len(val_dataset.warm_nodes)}")
    print(f"Test  - Nodes: {test_dataset.num_nodes}, Edges: {test_dataset.edge_index.shape[1]}, "
          f"Warm: {len(test_dataset.warm_nodes)}")
    print(f"Feature dimension: {train_dataset.node_features.shape[1]}")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing GNP model...")
    print("=" * 80)
    
    # Use max number of nodes across all splits
    max_nodes = max(train_dataset.num_nodes, val_dataset.num_nodes, test_dataset.num_nodes)
    
    model = GNP(
        num_nodes=max_nodes,
        feature_dim=train_dataset.node_features.shape[1],
        embedding_dim=200,
        num_layers=3,
        num_walks=25,
        dropout_ratio=0.5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Training model...")
    print("=" * 80)
    losses = train_gnp(model, train_dataset, num_epochs=100, batch_size=256, learning_rate=0.001)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Evaluating on validation set...")
    print("=" * 80)
    val_metrics = evaluate_gnp(model, val_dataset)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    test_metrics = evaluate_gnp(model, test_dataset)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Save model
    print("\n" + "=" * 80)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }, 'gnp_model.pt')
    print("Model saved to gnp_model.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()