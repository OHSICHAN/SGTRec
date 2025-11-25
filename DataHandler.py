import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, eye
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
from scipy.sparse.linalg import eigs

class DataHandler:
    def __init__(self):
        if args.data == 'yelp':
            predir = 'Data/yelp/'
        elif args.data == 'ml10m':
            predir = 'Data/ml10m/'
        elif args.data == 'tmall':
            predir = 'Data/tmall/'
        elif args.data == 'gowalla':
            predir = 'Data/gowalla/'
        elif args.data == 'amazon-book':
            predir = 'Data/amazon-book/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'
        self._L_eigs_low = None
        self._L_eigs_high = None
        self._indices = None

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret
    
    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    
    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)
    
        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def makeSample(self):
        user_sample_idx = t.tensor([[args.user + i for i in range(args.item)] * args.user])
        item_sample_idx = t.tensor([[i for i in range(args.user)] * args.item])
        return user_sample_idx, item_sample_idx
    
    def makeMask(self):
        u_u_mask = t.zeros(size=(args.user, args.user), dtype=bool)
        u_i_mask = t.ones(size=(args.user, args.item), dtype=bool)
    
        i_i_mask = t.zeros(size=(args.item, args.item), dtype=bool)
        i_u_mask = t.ones(size=(args.item, args.user), dtype=bool)
    
        u_mask = t.concat([u_u_mask, u_i_mask], dim=-1)
        i_mask = t.concat([i_u_mask, i_i_mask], dim=-1)
    
        mask = t.concat([u_mask, i_mask], dim=0)
        return mask
            
    def L_eigs(self):
        if self._L_eigs_low is None or self._L_eigs_high is None:
            if args.eigs_dim == 0:
                device = self.torchBiAdj.device
                self._L_eigs_low  = t.tensor([]).to(device)
                self._L_eigs_high = t.tensor([]).to(device)
            else:
                num_nodes = args.user + args.item
                A_sp = sp.csr_matrix(
                    (self.torchBiAdj._values().cpu().numpy(),
                    (self.torchBiAdj._indices()[0].cpu().numpy(),
                    self.torchBiAdj._indices()[1].cpu().numpy())),
                    shape=(num_nodes, num_nodes)
                )
                L_sp = sp.eye(num_nodes, format='csr') - A_sp
                # L_sp = I_sp - A_sp

                _, low_vecs = eigs(L_sp, k=args.eigs_dim, which='SR')
                _, high_vecs = eigs(L_sp, k=args.eigs_dim, which='LR')
                device = self.torchBiAdj.device

                self._L_eigs_low  = t.from_numpy(low_vecs.real).float().to(device)
                self._L_eigs_high = t.from_numpy(high_vecs.real).float().to(device)
        return self._L_eigs_low, self._L_eigs_high

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.mask = self.makeMask()
        
        low, high = self.L_eigs()
        self.eigen_low  = low
        self.eigen_high = high

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnMaskedData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
    
    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
    
    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
    
        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs
    
    def __len__(self):
        return len(self.tstUsrs)
    
    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])