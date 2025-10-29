import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_mean_pool as gap

#构建模型
#药物分子结构GCN
class Drug_Molecular(torch.nn.Module):
    def __init__(self, dim_drug, hidden_dim, output_dim, heads=4, dropout=0.2):
        super(Drug_Molecular,self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.drug_conv1 = GATConv(dim_drug, hidden_dim, heads=heads)# GCN卷积层
        self.drug_bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.drug_conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
        self.drug_bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, data , return_attention_weights=False):
        drug_feature, drug_adj, ibatch = data.x, data.edge_index, data.batch
        
        x_drug, attention_weights_1 = self.drug_conv1(drug_feature, drug_adj, return_attention_weights=True)
        x_drug = self.relu(x_drug)
        x_drug = self.drug_bn1(x_drug)
        x_drug = self.dropout(x_drug)
        x_drug, attention_weights_2 = self.drug_conv2(x_drug, drug_adj, return_attention_weights=True)
        x_drug = self.drug_bn2(x_drug)
        x_drug = gmp(x_drug, batch=ibatch)
        if return_attention_weights:
            return x_drug, (attention_weights_1, attention_weights_2)
        else:
            return x_drug

#细胞系表达降维
class Cell_Line(torch.nn.Module):
    def __init__(self, dim_cellline, hidden_dim, dropout=0.2):
        super(Cell_Line,self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.cell_fc1 = nn.Linear(dim_cellline, hidden_dim)
        self.cell_bn1 = nn.BatchNorm1d(hidden_dim)# 标准化

    def forward(self, gexpr_data):
        x_cell = self.cell_fc1(gexpr_data)
        x_cell = self.cell_bn1(x_cell)
        return x_cell

#GO相似性网络GCN
class GO_Network(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0.2):
        super(GO_Network,self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.GO_conv1 = GCNConv(feature_dim, hidden_dim)
        self.GO_bn1 = nn.BatchNorm1d(hidden_dim)
        self.GO_conv2 = GCNConv(hidden_dim, output_dim)
        self.GO_bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, GO_adj, GO_weight, drug_smiles_fea):
        x_GO = self.GO_conv1(drug_smiles_fea, GO_adj ,GO_weight)
        x_GO = self.GO_bn1(x_GO)
        x_GO = self.relu(x_GO)
        x_GO = self.dropout(x_GO)
        x_GO = self.GO_conv2(x_GO, GO_adj ,GO_weight)
        x_GO = self.GO_bn2(x_GO)     
        return x_GO

#ATC相似性网络GCN
class ATC_Network(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0.2):
        super(ATC_Network,self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.ATC_conv1 = GCNConv(feature_dim, hidden_dim)
        self.ATC_bn1 = nn.BatchNorm1d(hidden_dim)
        self.ATC_conv2 = GCNConv(hidden_dim, output_dim)
        self.ATC_bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, ATC_adj, ATC_weight, drug_smiles_fea):
        x_ATC = self.ATC_conv1(drug_smiles_fea, ATC_adj ,ATC_weight)
        x_ATC = self.ATC_bn1(x_ATC)
        x_ATC = self.relu(x_ATC)
        x_ATC = self.dropout(x_ATC)        
        x_ATC = self.ATC_conv2(x_ATC, ATC_adj ,ATC_weight)
        x_ATC = self.ATC_bn2(x_ATC)            
        return x_ATC

class CNN_Drug(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim,dropout=0.2):
        super(CNN_Drug, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.embed_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.embed_bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, drug_embed, cell_embed, index_drug, index_cell):
        embed = torch.cat((drug_embed[index_drug,:], cell_embed[index_cell,:]),1)
        embed = self.embed_fc1(embed)
        embed = self.embed_bn1(embed)
        return embed

class CNN_GO(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim,dropout=0.2):
        super(CNN_GO, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.embed_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.embed_bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, go_embed, cell_embed, index_drug, index_cell):
        embed = torch.cat((go_embed[index_drug,:], cell_embed[index_cell,:]),1)
        embed = self.embed_fc1(embed)
        embed = self.embed_bn1(embed)
        return embed

class CNN_ATC(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim,dropout=0.2):
        super(CNN_ATC, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.embed_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.embed_bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, atc_embed, cell_embed, index_drug, index_cell):
        embed = torch.cat((atc_embed[index_drug,:], cell_embed[index_cell,:]),1)
        embed = self.embed_fc1(embed)
        embed = self.embed_bn1(embed)
        return embed

class FCNN(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim,dropout=0.2):
        super(FCNN, self).__init__()
        self.last_layer_feature = None
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.embed_fc1 = nn.Linear(embed_dim, hidden_dim[0])
        self.embed_bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.embed_fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.embed_bn2 = nn.BatchNorm1d(hidden_dim[1])
        self.embed_fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.embed_bn3 = nn.BatchNorm1d(hidden_dim[2])
        self.embed_fc4 = nn.Linear(hidden_dim[2], 1)

    def forward(self, Drug1, Drug2, GO1, GO2, ATC1, ATC2, cell_embed, index):
        embed = torch.cat((Drug1, Drug2, GO1, GO2, ATC1, ATC2, cell_embed[index[:, 2],:]),1)
        embed = self.embed_fc1(embed)
        embed = self.embed_bn1(embed)
        embed = self.relu(embed)
        embed = self.dropout(embed) 
        embed = self.embed_fc2(embed)
        embed = self.embed_bn2(embed)
        embed = self.relu(embed)
        embed = self.dropout(embed) 
        embed = self.embed_fc3(embed)
        embed = self.embed_bn3(embed)
        embed = self.relu(embed)
        embed = self.dropout(embed) 
        embed = self.embed_fc4(embed)
        return embed


class Synergy(torch.nn.Module):
    def __init__(self, Drug_Molecular, Cell_Line, GO_Network, ATC_Network, CNN_Drug, CNN_GO, CNN_ATC, FCNN):
        super(Synergy, self).__init__()
        self.Drug_Molecular = Drug_Molecular
        self.Cell_Line = Cell_Line
        self.GO_Network = GO_Network
        self.ATC_Network = ATC_Network
        self.CNN_Drug = CNN_Drug
        self.CNN_GO = CNN_GO
        self.CNN_ATC = CNN_ATC
        self.FCNN = FCNN

    def forward(self,drug_set, cell_set ,GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea,index):
        
        for i ,drug in enumerate(drug_set,0):
            drug_embed = self.Drug_Molecular(drug)

        for i,cell in enumerate(cell_set,0):
            cell_embed = self.Cell_Line(cell)
            cell_exp = cell

        go_embed = self.GO_Network(GO_adj, GO_weight, drug_smiles_fea)
        atc_embed = self.ATC_Network(ATC_adj, ATC_weight, drug_smiles_fea)

        Drug1 = self.CNN_Drug(drug_embed, cell_embed, index[:, 0],index[:, 2])
        Drug2 = self.CNN_Drug(drug_embed, cell_embed, index[:, 1],index[:, 2])
        GO1 = self.CNN_GO(go_embed, cell_embed, index[:, 0],index[:, 2])
        GO2 = self.CNN_GO(go_embed, cell_embed, index[:, 1],index[:, 2])
        ATC1 = self.CNN_GO(atc_embed, cell_embed, index[:, 0],index[:, 2])
        ATC2 = self.CNN_GO(atc_embed, cell_embed, index[:, 1],index[:, 2])

        synergy_score = self.FCNN(Drug1, Drug2, GO1, GO2, ATC1, ATC2, cell_embed, index)
        
        feature = [drug_embed, cell_embed, go_embed, atc_embed]
        feature = [drug_embed, cell_embed, go_embed, atc_embed, torch.cat((Drug1,Drug2,GO1,GO2,ATC1,ATC2,cell_exp[index[:, 2],:]),1)]

        return synergy_score, feature
        

        