import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict
from Transformer import Encoder_Layer, TransformerEncoderLayer

class TransGNN(nn.Module):
    def __init__(self):
        super(TransGNN, self).__init__()

        self.user_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.user, args.latdim)))
        self.item_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.item, args.latdim)))
        
        self.eigen_low = None
        self.eigen_high = None

        self.mlp = nn.Linear(args.eigs_dim, args.latdim)

        self.user_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head, dropout=args.dropout)
        self.item_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head, dropout=args.dropout)

        self.hop_weights = nn.Parameter(torch.ones(args.block_num + 1))
        self.beta = nn.Parameter(torch.ones(1))
            
    def user_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.user_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.user_transformer_encoder(embeds, mask)
        
        return embeds
    
    def item_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.item_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.item_transformer_encoder(embeds, mask)
        
        return embeds
    
    def gnn_message_passing(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def forward(self, adj):
        init_embeds = torch.concat([self.user_embeding, self.item_embeding], dim=0)

        if self.eigen_low is not None and self.eigen_high is not None:
            low_emb = self.mlp(self.eigen_low)
            high_emb = self.mlp(self.eigen_high)
        else:
            low_emb = 0
            high_emb = 0

        embs = [init_embeds]
        
        for layer in range(args.block_num):
            h = embs[-1]
            
            total_eigen = low_emb + (self.beta**2) * high_emb
            h = h + total_eigen

            h = self.gnn_message_passing(adj, h)

            u_h = h[:args.user]
            i_h = h[args.user:]
            u_h = self.user_transformer_layer(u_h)
            i_h = self.item_transformer_layer(i_h)

            u_h = u_h + h[:args.user]
            i_h = i_h + h[args.user:]
            h = torch.concat([u_h, i_h], dim=0)

            embs.append(h)

        emb_stack = torch.stack(embs, dim=1)
        weights = F.softmax(self.hop_weights, dim=0)
        final_embeds = torch.einsum('nld,l->nd', emb_stack, weights)
        
        user_embeds = final_embeds[:args.user]
        item_embeds = final_embeds[args.user:]
        return final_embeds, user_embeds, item_embeds

    def pickEdges(self, adj):
        idx = adj._indices()
        rows, cols = idx[0, :], idx[1, :]
        mask = torch.logical_and(rows <= args.user, cols > args.user)
        rows, cols = rows[mask], cols[mask]
        edgeSampNum = int(args.edgeSampRate * rows.shape[0])
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = torch.randint(rows.shape[0], [edgeSampNum])
        pckUsrs, pckItms = rows[edgeids], cols[edgeids] - args.user
        return pckUsrs, pckItms
    
    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(args.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = torch.randint(args.user, [edgeSampNum])
        cols = torch.randint(args.item, [edgeSampNum])
        return rows, cols
    
    def bprLoss(self, user_embeding, item_embeding, ancs, poss, negs):
        ancEmbeds = user_embeding[ancs]
        posEmbeds = item_embeding[poss]
        negEmbeds = item_embeding[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss
    
    def calcLosses(self, ancs, poss, negs, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        user_embeding, item_embeding = embeds[:args.user], embeds[args.user:]

        bprLoss = self.bprLoss(user_embeding, item_embeding, ancs, poss, negs) + self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        return bprLoss
    
    def predict(self, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds
