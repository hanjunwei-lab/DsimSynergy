import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import os
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from model import Drug_Molecular, Cell_Line, GO_Network, ATC_Network, CNN_Drug, CNN_GO, CNN_ATC, FCNN, Synergy
from drug_util import GraphDataset, collate
from process_data import getData
from utils import  metric, set_seed_all, SynergyDataset
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_data(cell_exp_path,drug_synergy_path):
    """
    Load drug, cell line, and synergy data.
    Filter synergy scores: >10 as positive (label=1), <0 as negative (label=0).
    """
    drug_feature, drug_go_adj_weight, drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map = getData(cell_exp_path,drug_synergy_path)
    # Filter samples based on Loewe score
    filtered_synergy = []
    for row in synergy:
        if row[3] > 10:  # Positive if Loewe > 10
            row[3] = 1
            filtered_synergy.append(row)
        elif row[3] < 0:  # Negative if Loewe < 0
            row[3] = 0
            filtered_synergy.append(row)
    return drug_feature, drug_go_adj_weight, drug_atc_adj_weight, drug_smiles_fea, cell_feature, filtered_synergy, d_map, c_map

def data_split(synergy, rd_seed=0):
    """
    Split the synergy data into training/validation and test sets.
    Use stratified sampling to keep class distribution balanced.
    """

    columns = ['Drug1', 'Drug2', 'Cell_line', 'Loewe']
    synergy = pd.DataFrame(synergy, columns=columns)

    # Separate positive and negative samples
    synergy_pos = synergy[synergy['Loewe'] == 1]
    synergy_neg = synergy[synergy['Loewe'] == 0]
    train_size = 0.9

    # Shuffle and split positive samples
    synergy_cv_pos = synergy_pos.sample(frac=train_size, random_state=rd_seed)
    synergy_test_pos = synergy_pos.drop(synergy_cv_pos.index)
   
    #Shuffle and split negative samples
    synergy_cv_neg = synergy_neg.sample(frac=train_size, random_state=rd_seed)
    synergy_test_neg = synergy_neg.drop(synergy_cv_neg.index)  

    # Combine positive and negative splits and shuffle
    synergy_cv_data = pd.concat([synergy_cv_neg, synergy_cv_pos]).sample(frac=1, random_state=rd_seed).reset_index(drop=True)
    synergy_test = pd.concat([synergy_test_neg, synergy_test_pos]).sample(frac=1, random_state=rd_seed).reset_index(drop=True)

    return synergy_cv_data,synergy_test

# --train+test
def train(train_loader, drug_set, cell_set, GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea):
    """
    Train the model for one epoch.
    Returns evaluation metrics and average loss.
    """
    total_loss = 0.0
    compare = pd.DataFrame(columns=('prob','pred','true'))
    count = 0

    for i, data in enumerate(train_loader, 0):
        # Get indices and labels
        index, labels = data
        labels = labels.long()
        optimizer.zero_grad()

        # Forward pass
        pred_score, _ = model(drug_set, cell_set,GO_adj, GO_weight, 
                              ATC_adj, ATC_weight,drug_smiles_fea, index)
        # Compute loss
        loss = loss_func(pred_score.cpu(), labels.cpu())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute probabilities and predictions
        probs = F.softmax(pred_score, dim=1)
        probs = probs[:,1]
        _, pred = torch.max(pred_score.data, 1)

        # Move to CPU and convert to lists
        labels_list = labels.detach().cpu().numpy().tolist()
        pred_list = pred.detach().cpu().numpy().tolist()
        probs_list = probs.detach().cpu().numpy().tolist()

        compare_temp = pd.DataFrame({
            'prob': probs_list,
            'pred': pred_list,
            'true': labels_list
        })
        compare = pd.concat([compare,compare_temp])
        count += 1
        running_loss = 0.0

    # Compute evaluation metrics
    roc_auc, prc_auc, acc, bacc, precision, recall, kappa, f1 = metric(compare)

    return [roc_auc, prc_auc, acc, bacc, precision, recall, kappa, f1], total_loss/count 

def test(test_loader, drug_set, cell_set, GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea):
    """
    Evaluate the model on test or validation data.
    Returns evaluation metrics, average loss, features, and prediction results.
    """
    model.eval()
    total_loss = 0.0
    compare = pd.DataFrame(columns=('prob','pred','true'))
    synergy_data = pd.DataFrame(columns=('Drug1', 'Drug2', 'Cell_line', 'Loewe', 'Pred_Loewe',' Prob'))
    count = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            index, labels = data
            labels = labels.long()
            # Forward pass
            pred_score, feature = model(drug_set, cell_set ,GO_adj, GO_weight, 
                                        ATC_adj, ATC_weight,drug_smiles_fea,index)
            
            # Compute loss
            loss = loss_func(pred_score.cpu(), labels.cpu()) 
            total_loss += loss.item()
            
            # Predictions
            probs = F.softmax(pred_score, dim=1)
            probs = probs[:,1]
            _, pred = torch.max(pred_score.data, 1)
            
            # Convert to lists
            labels_list = labels.detach().cpu().numpy().tolist()
            pred_list = pred.detach().cpu().numpy().tolist()
            probs_list = probs.detach().cpu().numpy().tolist()

            drug1_list = index[:, 0].tolist()  # Drug1
            drug2_list = index[:, 1].tolist()  # Drug2
            cell_line_list = index[:, 2].tolist()  # Cell_line
            compare_temp = pd.DataFrame({
                'prob': probs_list,
                'pred': pred_list,
                'true': labels_list
            })
            synergy_temp = pd.DataFrame({
                'Drug1': drug1_list,
                'Drug2': drug2_list,
                'Cell_line': cell_line_list,
                'Loewe': labels_list,
                'Pred_Loewe': pred_list,
                'Prob': probs_list
            })
            compare = pd.concat([compare,compare_temp])
            synergy_data = pd.concat([synergy_data,synergy_temp])
            count += 1
        # Compute evaluation metrics
        roc_auc, prc_auc, acc, bacc, precision, recall, kappa, f1 = metric(compare)
        return [roc_auc, prc_auc, acc, bacc, precision, recall, kappa, f1], total_loss/count, feature, synergy_data
    

if __name__ == '__main__':
    cell_exp_path = '../Data/TRAIN/train_cell_exp.csv'
    drug_synergy_path = '../Data/TRAIN/train_synergy.csv'
    dataset_name = "TRAIN"
    seed = 42
    set_seed_all(seed)

    # Load data
    drug_feature , drug_go_adj_weight , drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map = load_data(cell_exp_path,drug_synergy_path)
    
    drug_smiles_fea = torch.tensor(drug_smiles_fea).to(device)
    # Reverse mappings for ID decoding
    reverse_d_map = {v: k for k, v in d_map.items()}
    reverse_c_map = {v: k for k, v in c_map.items()}
    # Move graph adjacency matrix and weights to device
    GO_adj = drug_go_adj_weight[0].to(device)
    GO_weight = drug_go_adj_weight[1].to(device)
    ATC_adj = drug_atc_adj_weight[0].to(device)
    ATC_weight = drug_atc_adj_weight[1].to(device)
    # Create data loaders for drug and cell line features
    cell_feature = torch.tensor(cell_feature, dtype=torch.float32).to(device)
    
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                            collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
    cell_set = Data.DataLoader(dataset=cell_feature,
                               batch_size=len(cell_feature), shuffle=False)
    
    # Split synergy data into cross-validation and test sets
    synergy_cv, synergy_test= data_split(synergy)
    final_metric = np.zeros(8)
    fold_num = 1 
    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in skf.split(synergy_cv.index, synergy_cv['Loewe']):
        # Ensure model directory exists
        model_dir = 'models' + str(fold_num)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Split into train and validation sets
        synergy_train = synergy_cv.iloc[train_index]
        synergy_validation = synergy_cv.iloc[validation_index]

        # Create data loaders
        train_loader = Data.DataLoader(SynergyDataset(synergy_train), batch_size=512, shuffle=True)
        val_loader = Data.DataLoader(SynergyDataset(synergy_validation), batch_size=2048, shuffle=True)
        test_loader = Data.DataLoader(SynergyDataset(synergy_test), batch_size=2048, shuffle=False)

        # ---model_build
        DM_dim = [75,512,256] # Drug_Molecular
        CL_dim = [len(cell_feature[1]),256] # Cell_Line
        GN_dim = [len(drug_smiles_fea[1]),512,256] # GO_Network
        AN_dim = [len(drug_smiles_fea[1]),512,256] # ATC_Network
        
        CD_dim = [512,256] #Drug_Molecular+Cell_Line
        CO_dim = [512,256] #GO_Network+Cell_Line
        CT_dim = [512,256] #ATC_Network+Cell_Line
        FN_dim = [(CD_dim[1] * 2 + CO_dim[1] * 2 + CT_dim[1]* 2 + CL_dim[1]),[1024,512,128]]

        model = Synergy(Drug_Molecular(dim_drug = DM_dim[0], hidden_dim = DM_dim[1], output_dim = DM_dim[2], heads=4),
                        Cell_Line(dim_cellline = CL_dim[0], hidden_dim = CL_dim[1]),
                        GO_Network(feature_dim = GN_dim[0], hidden_dim = GN_dim[1], output_dim = GN_dim[2]),
                        ATC_Network(feature_dim = AN_dim[0], hidden_dim = AN_dim[1], output_dim = AN_dim[2]),
                        CNN_Drug(embed_dim = CD_dim[0], hidden_dim = CD_dim[1]),
                        CNN_GO(embed_dim = CO_dim[0], hidden_dim = CO_dim[1]),
                        CNN_ATC(embed_dim = CT_dim[0], hidden_dim = CT_dim[1]),
                        FCNN(embed_dim = FN_dim[0], hidden_dim = FN_dim[1])
                        ).to(device)
        
        
        epochs = 1000
        learning_rate = 0.0001
        # Define weighted loss for imbalanced classes
        weights = torch.tensor([0.5, 16], dtype=torch.float32)
        loss_func = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), weight_decay = 1e-4, amsgrad = False)
        
        # Training tracking
        train_losses = []
        val_losses = []

        # ---run
        best_metric = np.zeros(8)
        best_epoch = 0
        best_loss = float('inf') 
        for epoch in range(epochs):
            model.train()
            train_metric, train_loss = train(train_loader, drug_set, cell_set,
                                             GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea
                                             )
            val_metric, val_loss, _, _ = test(val_loader, drug_set, cell_set,
                                           GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea
                                             )
            torch.cuda.empty_cache()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                  'AUROC: {:.6f},'.format(train_metric[0]), 'AUPR: {:.6f},'.format(train_metric[1]),
                  'ACC: {:.6f},'.format(train_metric[2]),'BACC: {:.6f},'.format(train_metric[3]),
                  'PRECISION: {:.6f},'.format(train_metric[4]),'RECALL: {:.6f},'.format(train_metric[5]),
                  'KAPPA: {:.6f},'.format(train_metric[6]),'F1: {:.6f},'.format(train_metric[7])
                 )
            print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                  'AUROC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                  'ACC: {:.6f},'.format(val_metric[2]),'BACC: {:.6f},'.format(val_metric[3]),
                  'PRECISION: {:.6f},'.format(val_metric[4]),'RECALL: {:.6f},'.format(val_metric[5]),
                  'KAPPA: {:.6f},'.format(val_metric[6]),'F1: {:.6f},'.format(val_metric[7])
                 )
            # Save model at each epoch
            torch.save(model.state_dict(), os.path.join(model_dir, '{}.pth'.format(epoch)))
            # Update best model based on F1 score
            if val_metric[7] > best_metric[7]:
                best_metric = val_metric
                best_epoch = epoch
            # Remove outdated model checkpoints
            files = glob.glob(os.path.join(model_dir, '*.pth'))
            for f in files:
                f = os.path.basename(f)
                epoch_nb = int(f.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(os.path.join(model_dir,f))

        # Save training/validation loss curves
        loss_df = pd.DataFrame({'Epoch': range(len(train_losses)), 'Train Loss': train_losses, 'Validation Loss': val_losses})
        loss_df.to_csv(model_dir + "/" + dataset_name + "_" + str(fold_num) + '_loss_values.csv', index=False)
        # Clean up models after best epoch
        files = glob.glob(os.path.join(model_dir, '*.pth'))
        for f in files:
            f = os.path.basename(f)
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(os.path.join(model_dir,f))
        # Reload best model
        model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(best_epoch))))
        torch.cuda.empty_cache()
        # Evaluate on train, validation, and test sets
        train_metric, _, _, synergy_train_rev = test(train_loader, drug_set, cell_set,
                                                    GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                                    )
        val_metric, _, _, synergy_validation_rev = test(val_loader, drug_set, cell_set,
                                         GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                         )
        test_metric, _, _, synergy_test_rev = test(test_loader, drug_set, cell_set,
                                            GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                            ) 
        
        # Map indices back to original IDs
        synergy_test_rev['Drug1'] = synergy_test_rev['Drug1'].map(reverse_d_map)
        synergy_test_rev['Drug2'] = synergy_test_rev['Drug2'].map(reverse_d_map)
        synergy_test_rev['Cell_line'] = synergy_test_rev['Cell_line'].map(reverse_c_map) 
        synergy_test_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe','Prob']].to_csv(model_dir + "/" + dataset_name + '_test_' +  str(fold_num) + '_pred.csv', index=False, sep=',')

        synergy_train_rev['Drug1'] = synergy_train_rev['Drug1'].map(reverse_d_map)
        synergy_train_rev['Drug2'] = synergy_train_rev['Drug2'].map(reverse_d_map)
        synergy_train_rev['Cell_line'] = synergy_train_rev['Cell_line'].map(reverse_c_map) 
        synergy_train_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe','Prob']].to_csv(model_dir + "/" + dataset_name + '_train_' + str(fold_num) + '_pred.csv', index=False, sep=',')
        
        synergy_validation_rev['Drug1'] = synergy_validation_rev['Drug1'].map(reverse_d_map)
        synergy_validation_rev['Drug2'] = synergy_validation_rev['Drug2'].map(reverse_d_map)
        synergy_validation_rev['Cell_line'] = synergy_validation_rev['Cell_line'].map(reverse_c_map) 
        synergy_validation_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe','Prob']].to_csv(model_dir + "/" + dataset_name + '_val_' + str(fold_num) + '_pred.csv', index=False, sep=',')
        # Accumulate test metrics
        final_metric += test_metric
        fold_num = fold_num + 1
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
              'AUROC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
              'ACC: {:.6f},'.format(val_metric[2]),'BACC: {:.6f},'.format(val_metric[3]),
              'PRECISION: {:.6f},'.format(val_metric[4]),'RECALL: {:.6f},'.format(val_metric[5]),
              'KAPPA: {:.6f},'.format(val_metric[6]),'F1: {:.6f},'.format(val_metric[7])
             )
        print('The best results on test set, Epoch: {:05d},'.format(best_epoch),
              'AUROC: {:.6f},'.format(test_metric[0]), 'AUPR: {:.6f},'.format(test_metric[1]),
              'ACC: {:.6f},'.format(test_metric[2]),'BACC: {:.6f},'.format(test_metric[3]),
              'PRECISION: {:.6f},'.format(test_metric[4]),'RECALL: {:.6f},'.format(test_metric[5]),
              'KAPPA: {:.6f},'.format(test_metric[6]),'F1: {:.6f},'.format(test_metric[7])
              )
    # Output final average metrics over 5 folds
    final_metric /= 5
    print('Final 5-cv average results,',
          'AUROC: {:.6f},'.format(final_metric[0]), 'AUPR: {:.6f},'.format(final_metric[1]),
          'ACC: {:.6f},'.format(final_metric[2]),'BACC: {:.6f},'.format(final_metric[3]),
          'PRECISION: {:.6f},'.format(final_metric[4]),'RECALL: {:.6f},'.format(final_metric[5]),
          'KAPPA: {:.6f},'.format(final_metric[6]),'F1: {:.6f},'.format(final_metric[7])
         )