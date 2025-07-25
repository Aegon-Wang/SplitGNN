import torch
import dgl
import copy
import sympy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import hinge_loss
import scipy
import dgl.function as fn
import math

"""
Cooperative Attention Mechanism Integration

This module integrates a Cooperative Attention mechanism into the SplitGNN model to enhance
robustness and discriminability by combining structural neighbors and feature neighbors.

Key Features:
1. Structural Attention: Computes attention weights based on graph topology/structure
2. Feature Attention: Computes attention weights based on feature similarity using cosine similarity
3. Cooperative Fusion: Combines both attention types through a learnable gating mechanism
4. Residual Connection: Maintains gradient flow and model stability
5. Layer Normalization: Stabilizes training and improves convergence

The cooperative attention addresses the biases that come from relying on only one type of
neighbor information:
- Structural neighbors can be biased by graph topology
- Feature neighbors can be biased by feature similarity
- Cooperative attention mitigates these biases through mutual collaboration

Usage:
- Set 'use_cooperative_attention: True' in config files to enable
- The mechanism is integrated into MultiRelationSplitGNNLayer
- Attention is applied before the PolyConv operations
"""

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 relation_aware,
                 thetas,
                 K,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=True):
        super(PolyConv, self).__init__()
        self._theta = thetas
        self._k = len(self._theta[0])
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.relation_aware = relation_aware
        self.K = K
        self.linear = nn.Linear(in_feats*len(thetas), out_feats, bias)
        self.linear1 = nn.Linear(in_feats*len(thetas), out_feats, bias)
        self.transh = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph, flag):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            if flag==0:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            elif flag==1:
                graph.update_all(self.message_positive, fn.sum('p', 'h'))
            elif flag==2:
                graph.update_all(self.message_negative, fn.sum('n', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(self.sign_edges)
            graph.apply_edges(self.judge_edges)
            
            graph.update_all(message_func=fn.copy_e('positive', 'positive'), reduce_func=self.positive_reduce)
            graph.update_all(message_func=fn.copy_e('negative', 'negative'), reduce_func=self.negative_reduce)

            positive_in_degrees = graph.ndata['positive_in_degree']
            negative_in_degrees = graph.ndata['negative_in_degree']
            
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_positive = torch.pow(positive_in_degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_negative = torch.pow(negative_in_degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            
            hs_o = []
            hs_p = []
            hs_n = []
            
            transh = self.transh(feat)
            
            for theta in self._theta:
                h_o = theta[0]*feat
                
                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt, graph, 0)
                    h_o += theta[k]*feat
                hs_o.append(h_o)
            
            feat = graph.ndata['feat']
            for theta in self._theta[0:self.K+1]:
                h_p = theta[0]*feat
                
                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt_positive, graph, 1)
                    h_p += theta[k]*feat
                hs_p.append(h_p)

            feat = graph.ndata['feat']
            for theta in self._theta[self.K+1:]:
                h_n = theta[0]*feat

                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt_negative, graph, 2)
                    h_n += theta[k]*feat
                hs_n.append(h_n)
        
            hs_o = torch.cat(hs_o, dim=1)
            if self.K != len(self._theta) - 1 and self.K != -1:
                hs_p = torch.cat(hs_p, dim=1)
                hs_n = torch.cat(hs_n, dim=1) 
                hs_pn = torch.cat([hs_p, hs_n], dim=1)
            elif self.K == -1:
                hs_pn = torch.cat(hs_n, dim=1)
            else:
                hs_pn = torch.cat(hs_p, dim=1)

        if self.lin:
            hs_o = self.linear(hs_o)
            hs_o = self.activation(hs_o)
            hs_pn = self.linear1(hs_pn)
            hs_pn = self.activation(hs_pn)

        return hs_o, hs_pn, transh

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'sign':torch.sign(score)}

    def judge_edges(self, edges):
        return {'positive': (edges.data['sign'] >= 0).float(), 'negative': (edges.data['sign'] < 0).float()}

    def positive_reduce(self, nodes):
        return {'positive_in_degree': nodes.mailbox['positive'].sum(1)}

    def negative_reduce(self, nodes):
        return {'negative_in_degree': nodes.mailbox['negative'].sum(1)}
    
    def message_positive(self, edges):
        mask = (edges.data['sign'] >= 0).float().view(-1, 1)
        masked_src_feats = edges.src['h'] * mask
        return {'p': masked_src_feats}
    
    def message_negative(self, edges):
        mask = (edges.data['sign'] < 0).float().view(-1, 1)
        masked_src_feats = edges.src['h'] * mask
        return {'n': masked_src_feats}
    
    
class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3*output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


class CooperativeAttention(nn.Module):
    """
    Cooperative Attention mechanism that combines structural neighbors and feature neighbors
    to enhance robustness and discriminability through mutual collaboration.
    """
    def __init__(self, in_feats, hidden_dim=64, dropout=0.1):
        super(CooperativeAttention, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Structural attention components
        self.struct_query = nn.Linear(in_feats, hidden_dim, bias=False)
        self.struct_key = nn.Linear(in_feats, hidden_dim, bias=False)
        self.struct_value = nn.Linear(in_feats, hidden_dim, bias=False)
        
        # Feature attention components  
        self.feat_query = nn.Linear(in_feats, hidden_dim, bias=False)
        self.feat_key = nn.Linear(in_feats, hidden_dim, bias=False)
        self.feat_value = nn.Linear(in_feats, hidden_dim, bias=False)
        
        # Cooperative fusion components
        self.cooperative_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_feats)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(in_feats)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for module in [self.struct_query, self.struct_key, self.struct_value,
                      self.feat_query, self.feat_key, self.feat_value,
                      self.cooperative_gate, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def structural_attention(self, graph, feat):
        """Compute structural attention based on graph topology"""
        with graph.local_scope():
            # Transform features for attention computation
            query = self.struct_query(feat)
            key = self.struct_key(feat)
            value = self.struct_value(feat)
            
            graph.ndata['query'] = query
            graph.ndata['key'] = key
            graph.ndata['value'] = value
            
            # Message passing function for structural attention
            def struct_message_func(edges):
                # Compute attention scores based on structural relationships
                scores = (edges.src['query'] * edges.dst['key']).sum(dim=-1, keepdim=True)
                scores = scores / math.sqrt(self.hidden_dim)
                return {'score': scores, 'value': edges.src['value']}
            
            def struct_reduce_func(nodes):
                # Apply softmax attention and aggregate
                attention_weights = F.softmax(nodes.mailbox['score'], dim=1)
                attended_values = (attention_weights * nodes.mailbox['value']).sum(dim=1)
                return {'struct_att': attended_values}
            
            graph.update_all(struct_message_func, struct_reduce_func)
            struct_attention = graph.ndata['struct_att']
            
            return struct_attention
    
    def feature_attention(self, graph, feat):
        """Compute feature attention based on feature similarity"""
        with graph.local_scope():
            # Transform features for attention computation
            query = self.feat_query(feat)
            key = self.feat_key(feat)
            value = self.feat_value(feat)
            
            graph.ndata['query'] = query
            graph.ndata['key'] = key
            graph.ndata['value'] = value
            
            # Message passing function for feature attention
            def feat_message_func(edges):
                # Compute attention scores based on feature similarity
                # Use cosine similarity for feature-based attention
                query_norm = F.normalize(edges.dst['query'], p=2, dim=-1)
                key_norm = F.normalize(edges.src['key'], p=2, dim=-1)
                scores = (query_norm * key_norm).sum(dim=-1, keepdim=True)
                return {'score': scores, 'value': edges.src['value']}
            
            def feat_reduce_func(nodes):
                # Apply softmax attention and aggregate
                attention_weights = F.softmax(nodes.mailbox['score'], dim=1)
                attended_values = (attention_weights * nodes.mailbox['value']).sum(dim=1)
                return {'feat_att': attended_values}
            
            graph.update_all(feat_message_func, feat_reduce_func)
            feat_attention = graph.ndata['feat_att']
            
            return feat_attention
    
    def cooperative_fusion(self, struct_att, feat_att):
        """Cooperatively fuse structural and feature attention"""
        # Concatenate both attention representations
        combined = torch.cat([struct_att, feat_att], dim=-1)
        
        # Apply gating mechanism for cooperative fusion
        gate = torch.sigmoid(self.cooperative_gate(combined))
        
        # Weighted combination of structural and feature attention
        fused_attention = gate * struct_att + (1 - gate) * feat_att
        
        return fused_attention
    
    def forward(self, graph, feat):
        """
        Forward pass of cooperative attention
        Args:
            graph: DGL graph
            feat: node features [N, in_feats]
        Returns:
            Enhanced node features with cooperative attention
        """
        # Store original features for residual connection
        residual = feat
        
        # Compute structural attention (topology-based)
        struct_att = self.structural_attention(graph, feat)
        
        # Compute feature attention (similarity-based)
        feat_att = self.feature_attention(graph, feat)
        
        # Cooperatively fuse both attentions
        fused_att = self.cooperative_fusion(struct_att, feat_att)
        
        # Project back to original feature space
        output = self.output_proj(fused_att)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        return output


class MultiRelationSplitGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dataset, dropout, thetas, K, if_sum=False, use_cooperative_attention=True):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove('homo')
        # self.relation = ['homo']
        self.n_relation = len(self.relation)
        self.liner = nn.Linear(self.n_relation*output_dim*3, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        
        # Add Cooperative Attention mechanism
        self.use_cooperative_attention = use_cooperative_attention
        if self.use_cooperative_attention:
            self.cooperative_attention = CooperativeAttention(input_dim, hidden_dim=output_dim//2, dropout=dropout)
        
        for e in self.relation:
            self.minelayers[e] = PolyConv(input_dim, output_dim, self.relation_aware, thetas, K, lin=True)

    def forward(self, g, h):
        # Apply Cooperative Attention mechanism before processing
        if self.use_cooperative_attention:
            h = self.cooperative_attention(g, h)
        
        hs_o = []
        hs_lh = []
        hs_trans = []
        for e in self.relation:
            h_o, h_lh, h_trans = self.minelayers[e](g.edge_type_subgraph([e]), h)
            hs_o.append(h_o)
            hs_lh.append(h_lh)
            hs_trans.append(h_trans)
        h = torch.cat([torch.cat(hs_o, dim=1),torch.cat(hs_lh, dim=1), torch.cat(hs_trans, dim=1)], dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h
    
    def loss(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            agg_h = self.forward(g, h)

            g.apply_edges(self.score_edges, etype='homo')
            edges_score = g.edges['homo'].data['score']
            edge_train_mask = g.edges['homo'].data['train_mask'].bool()
            edge_train_label = g.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            return agg_h, edge_diff_loss
            
    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}


class SplitGNN(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        self.input_dim = g.nodes['r'].data['feature'].shape[1]  # nodes['company'] for FDCompCN
        self.intra_dim = args.intra_dim
        self.gamma = args.gamma
        self.C = args.C
        self.K = args.K
        self.n_class = args.n_class
        self.thetas = calculate_theta2(d=self.C)
        
        # Add cooperative attention option (default enabled)
        use_cooperative_attention = getattr(args, 'use_cooperative_attention', True)
        
        self.mine_layer = MultiRelationSplitGNNLayer(self.intra_dim, self.intra_dim, g, args.dropout, self.thetas, self.K, use_cooperative_attention=use_cooperative_attention)
        self.linear = nn.Linear(self.input_dim, self.intra_dim)
        self.linear2 = nn.Linear(self.intra_dim, self.n_class)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()

    def forward(self, g):
        feats = g.ndata['feature'].float()
        h = self.linear(feats)
        h = self.mine_layer(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return h 
    
    def loss(self, g):
        feats = g.ndata['feature'].float()
        h = self.linear(feats)
        h, edge_loss = self.mine_layer.loss(g, h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma*edge_loss
        return loss




