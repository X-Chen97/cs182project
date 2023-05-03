from __future__ import print_function, division

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class WeightShareConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(WeightShareConvLayer, self).__init__()
        ############################################################################
        # TODO: Build a WeightShareConvLayer according to equation explained in 
        # Notebook. Note that as a gate function we will use batchnormalization first
        # and then apply softplus. 
        # Hint: Try to undersatnd ConvLayer first
        
        # Solution
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_nbr = nn.Linear(
            self.atom_fea_len + self.nbr_fea_len, self.atom_fea_len
        )
        self.fc_in = nn.Linear(self.atom_fea_len, self.atom_fea_len)
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(self.atom_fea_len)
        ############################################################################

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        # convolution
        ############################################################################
        # TODO: Build a WeightShareConvLayer according to equation explained in 
        # Notebook. Note that as a gate function we will use batchnormalization first
        # and then apply softplus. 
        # Hint: Try to undersatnd ConvLayer first
        
        # Solution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        nbr_fc_fea = self.fc_nbr(total_nbr_fea)
        atom_in_fea = self.fc_in(atom_in_fea)
        total_fea = torch.sum(nbr_fc_fea, dim=1) + atom_in_fea
        total_gated_fea = self.bn(total_fea)
        out = self.softplus(total_gated_fea)
        ############################################################################

        return out


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        option='C'
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        #
        #     Input
        #       |
        #  - - - - - -        - - - - - - - -        - - - - -        - - - - - -
        # |           |      |               |      |         |      |           |
        # | Embedding | ---- | Conv. Layers  | ---- |  Linear | ---- |  Softplus |
        # |           |      |               |      |         |      |           |
        #  - - - - - -        - - - - - - - -        - - - - -        - - - - - -

        ############################################################################
        # TODO: Initialize the parameters.
        # 1. self.embedding: the embedding layer (orig_atom_fea_len x atom_fea_len)
        # 2. self.convs: list of convolutioal layers we built above. 
        #    Try to use ConvLayer with option "C" and use WeightShareConvLayer with
        #    option "WC". Hint: try using nn.ModuleList
        # 3. self.conv_to_fc: the last layer (atom_fea_len x h_fea_len)
        # 4. self.conv_to_fc_softplus: the softplus layer.
        
        # raise NotImplementedError()
        
        # Solution
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        if option == 'C':
            self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                        nbr_fea_len=nbr_fea_len)
                                        for _ in range(n_conv)])
        elif option == 'WC':
            self.convs = nn.ModuleList([WeightShareConvLayer(atom_fea_len=atom_fea_len,
                            nbr_fea_len=nbr_fea_len)
                            for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        ############################################################################
        
        # Hidden layers.
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        ############################################################################
        # TODO: Implement the forward function.
        # 1. Apply embedding to atom features
        # 2. Apply each convolution layer to the atom features.
        # 3. Applying pooling.
        # 4. Apply conv_to_fc
        # 5. Apply softplus
        
        # raise NotImplementedError()
        
        # Solution.
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        ############################################################################

        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        # Pooling function. What if we use torch.max here instead?
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)
