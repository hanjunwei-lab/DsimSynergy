import numpy as np
import pandas as pd         
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import os
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from model import Drug_Molecular, Cell_Line, GO_Network, ATC_Network, CNN_Drug, CNN_GO, CNN_ATC, FCNN, FCNN, Synergy
from drug_util import GraphDataset, collate
from process_data import getData
from utils import  metric, set_seed_all, SynergyDataset
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_data(cell_exp_path,drug_synergy_path):
    """
    Load drug features, cell line expression, and synergy data.
    Returns raw components including adjacency matrices, mappings, etc.
    """
    drug_feature, drug_go_adj_weight, drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map = getData(cell_exp_path,drug_synergy_path)
    return drug_feature, drug_go_adj_weight, drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map

def data_split(synergy, rd_seed=0):
    """
    Split the synergy dataset into training/validation (90%) and test set (10%).
    No stratification since it's a regression task.
    """
    columns = ['Drug1', 'Drug2', 'Cell_line', 'Loewe']  # 假设列名为这些
    synergy = pd.DataFrame(synergy, columns=columns)

    train_size = 0.9
    # Randomly split into training/validation set and test set
    synergy_cv_data = synergy.sample(frac=train_size, random_state=rd_seed)
    synergy_test = synergy.drop(synergy_cv_data.index)
    return synergy_cv_data, synergy_test

# --train+test
def train(train_loader, drug_set, cell_set, GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea):
    """
    Train the model for one epoch.
    Returns evaluation metrics and average loss.
    """    
    total_loss = 0.0
    compare = pd.DataFrame(columns=('pred','true'))
    count = 0
    for i, data in enumerate(train_loader, 0):
        index, labels = data
        labels = labels.reshape(labels.shape[0], 1).to(device)
        optimizer.zero_grad()
        pred_score,_ = model(drug_set, cell_set,GO_adj, GO_weight, 
                        ATC_adj, ATC_weight,drug_smiles_fea,index)
        
        loss = loss_func(pred_score.cpu(), labels.cpu())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1

        # Convert tensors to lists
        labels_list = labels[:,0].detach().cpu().numpy().tolist()
        pred_list = pred_score[:,0].detach().cpu().numpy().tolist()
        
        compare_temp = pd.DataFrame({
            'pred': pred_list,
            'true': labels_list
        })
        compare = pd.concat([compare,compare_temp])
    mse, rmse, r2, pear = metric(compare)
    return [mse, rmse, r2, pear],total_loss/count

    
def test(test_loader, drug_set, cell_set, GO_adj, GO_weight, ATC_adj, ATC_weight, drug_smiles_fea):
    """
    Evaluate the model on test or validation data.
    Returns metrics, average loss, predicted results, and extracted features.
    """
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    synergy_data = pd.DataFrame(columns=('Drug1', 'Drug2', 'Cell_line', 'Loewe', 'Pred_Loewe'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in test_loader:
            index, labels = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            pred_score,feature = model(drug_set, cell_set,GO_adj, GO_weight, 
                                       ATC_adj, ATC_weight,drug_smiles_fea,index)
            loss = loss_func(pred_score.cpu(), labels.cpu())
            total_loss += loss.item()
            count += 1

            labels_list = labels[:,0].detach().cpu().numpy().tolist()
            pred_list = pred_score[:,0].detach().cpu().numpy().tolist()
            
            # Split index into three parts: Drug1, Drug2, Cell_line
            drug1_list = index[:, 0].tolist()
            drug2_list = index[:, 1].tolist()
            cell_line_list = index[:, 2].tolist()
            compare_temp = pd.DataFrame({
            'pred': pred_list,
            'true': labels_list
            })
            synergy_temp = pd.DataFrame({
                'Drug1': drug1_list,
                'Drug2': drug2_list,
                'Cell_line': cell_line_list,
                'Loewe': labels_list,
                'Pred_Loewe': pred_list,
            })
            compare = pd.concat([compare,compare_temp])
            synergy_data = pd.concat([synergy_data,synergy_temp])
    mse, rmse, r2, pear = metric(compare)
    return [mse, rmse, r2, pear], total_loss/count, synergy_data, feature



if __name__ == '__main__':    
    cell_exp_path = '../Data/TRAIN/train_cell_exp.csv'
    drug_synergy_path = '../Data/TRAIN/train_synergy.csv'
    dataset_name = "TRAIN"

    seed = 42
    set_seed_all(seed)

    # Load all required data
    drug_feature , drug_go_adj_weight , drug_atc_adj_weight, drug_smiles_fea, cell_feature, synergy, d_map, c_map = load_data(cell_exp_path,drug_synergy_path)
    drug_smiles_fea = torch.tensor(drug_smiles_fea).to(device)
    
    # Reverse mapping from index to original IDs
    reverse_d_map = {v: k for k, v in d_map.items()}
    reverse_c_map = {v: k for k, v in c_map.items()}
    
    # Move graph structures and weights to device
    GO_adj = drug_go_adj_weight[0].to(device)
    GO_weight = drug_go_adj_weight[1].to(device)
    ATC_adj = drug_atc_adj_weight[0].to(device)
    ATC_weight = drug_atc_adj_weight[1].to(device)
    cell_feature = torch.tensor(cell_feature, dtype=torch.float32).to(device)
    
    # Create data loaders for precomputed features
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                            collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
    cell_set = Data.DataLoader(dataset=cell_feature,
                               batch_size=len(cell_feature), shuffle=False)
    
    # Split synergy data into cross-validation (train/val) and held-out test set
    synergy_cv, synergy_test= data_split(synergy)
    final_metric = np.zeros(4) # 初始化用于存储评价指标的数组
    fold_num = 1 
    # 5-fold CV without stratification
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in kf.split(synergy_cv.index):
        # Ensure model directory exists per fold
        model_dir = 'models' + str(fold_num)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        synergy_train = synergy_cv.iloc[train_index]
        synergy_validation = synergy_cv.iloc[validation_index]

        train_loader = Data.DataLoader(SynergyDataset(synergy_train), batch_size=512, shuffle=True)
        val_loader = Data.DataLoader(SynergyDataset(synergy_validation), batch_size=2048, shuffle=False)
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
        
        # Initialize the full synergy prediction model
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
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), weight_decay = 1e-4, amsgrad = False)
        
        # Lists to store loss values during training
        train_losses = []
        val_losses = []

        # ---run
        best_metric = np.zeros(4)
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

            # Record losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                  'MSE: {:.6f},'.format(train_metric[0]), 'RMSE: {:.6f},'.format(train_metric[1]),
                  'R2: {:.6f},'.format(train_metric[2]),'PEARSON: {:.6f},'.format(train_metric[3])
                 )
            print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                  'MSE: {:.6f},'.format(val_metric[0]), 'RMSE: {:.6f},'.format(val_metric[1]),
                  'R2: {:.6f},'.format(val_metric[2]),'PEARSON: {:.6f},'.format(val_metric[3])
                 )
            # Update best model based on Pearson correlation
            torch.save(model.state_dict(), os.path.join(model_dir, '{}.pth'.format(epoch)))
            if val_metric[3] > best_metric[3]:
                best_metric = val_metric
                best_epoch = epoch
            # Remove outdated checkpoints before best epoch     
            files = glob.glob(os.path.join(model_dir, '*.pth'))
            for f in files:
                f = os.path.basename(f)
                epoch_nb = int(f.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(os.path.join(model_dir,f))

        # Save training/validation loss history
        loss_df = pd.DataFrame({'Epoch': range(len(train_losses)), 'Train Loss': train_losses, 'Validation Loss': val_losses})
        loss_df.to_csv(model_dir + "/" + dataset_name + "_" + str(fold_num) + '_loss_values.csv', index=False)
        # Clean up checkpoints after best epoch
        files = glob.glob(os.path.join(model_dir, '*.pth'))
        for f in files:
            f = os.path.basename(f)
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(os.path.join(model_dir,f))
        # Load the best-performing model
        model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(best_epoch))))
        torch.cuda.empty_cache()
        # Perform final inference
        train_metric, _, synergy_train_rev, _= test(train_loader, drug_set, cell_set,
                                                    GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                                    )
        val_metric, _, synergy_validation_rev, _ = test(val_loader, drug_set, cell_set,
                                         GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                         )
        test_metric, _, synergy_test_rev, _ = test(test_loader, drug_set, cell_set,
                                            GO_adj, GO_weight, ATC_adj, ATC_weight,drug_smiles_fea
                                            ) 
        
        # Map back indices to original IDs
        synergy_test_rev['Drug1'] = synergy_test_rev['Drug1'].map(reverse_d_map)
        synergy_test_rev['Drug2'] = synergy_test_rev['Drug2'].map(reverse_d_map)
        synergy_test_rev['Cell_line'] = synergy_test_rev['Cell_line'].map(reverse_c_map) 
        synergy_test_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe']].to_csv(model_dir + "/" + dataset_name + '_test_' +  str(fold_num) + '_pred.csv', index=False, sep=',')

        synergy_train_rev['Drug1'] = synergy_train_rev['Drug1'].map(reverse_d_map)
        synergy_train_rev['Drug2'] = synergy_train_rev['Drug2'].map(reverse_d_map)
        synergy_train_rev['Cell_line'] = synergy_train_rev['Cell_line'].map(reverse_c_map) 
        synergy_train_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe']].to_csv(model_dir + "/" + dataset_name + '_train_' + str(fold_num) + '_pred.csv', index=False, sep=',')

        synergy_validation_rev['Drug1'] = synergy_validation_rev['Drug1'].map(reverse_d_map)
        synergy_validation_rev['Drug2'] = synergy_validation_rev['Drug2'].map(reverse_d_map)
        synergy_validation_rev['Cell_line'] = synergy_validation_rev['Cell_line'].map(reverse_c_map) 
        synergy_validation_rev[['Drug1', 'Drug2', 'Cell_line', 'Loewe','Pred_Loewe']].to_csv(model_dir + "/" + dataset_name + '_val_' + str(fold_num) + '_pred.csv', index=False, sep=',')
        final_metric += test_metric
        fold_num = fold_num + 1
        # Print best performance on validation and test sets
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
              'MSE: {:.6f},'.format(val_metric[0]), 'RMSE: {:.6f},'.format(val_metric[1]),
              'R2: {:.6f},'.format(val_metric[2]),'PEARSON: {:.6f},'.format(val_metric[3])
             )
        print('The best results on test set, Epoch: {:05d},'.format(best_epoch),
              'MSE: {:.6f},'.format(test_metric[0]), 'RMSE: {:.6f},'.format(test_metric[1]),
              'R2: {:.6f},'.format(test_metric[2]),'PEARSON: {:.6f},'.format(test_metric[3])
              )
    # Compute and print average performance over 5 folds
    final_metric /= 5
    print('Final 5-cv average results,',
          'MSE: {:.6f},'.format(final_metric[0]), 'RMSE: {:.6f},'.format(final_metric[1]),
          'R2: {:.6f},'.format(final_metric[2]),'PEARSON: {:.6f},'.format(final_metric[3])
         )