# Load required libraries
library(dplyr)
library(tidyr)
# Read drug-target interactions (human only, species ID 9606)
drug_target <- read.csv("Data/drug_target_9606.csv")
drug_target <- drug_target[which(drug_target$gene_name!=""),]
drug_target <- drug_target[,c(1,7)]
drug_target <- drug_target[!duplicated(drug_target),]
drug_target <- drug_target[order(drug_target$drugbank_id),]
# Read drug-ATC code mappings
drug_atc <- read.csv("Data/drug_atc.csv")
drug_atc <- drug_atc[which(drug_atc$atc_code!=""),]
# Read SMILES strings for drugs
drugbank_smiles <- data.table::fread("Data/drug_smiles.csv")
drugbank_smiles <- drugbank_smiles[which(drugbank_smiles$smiles!=''),]
# Find common drug IDs present in all three datasets: target, ATC, and SMILES
drug_id <- intersect(intersect(unique(drug_target$drugbank_id),unique(drug_atc$drugbank_id)),drugbank_smiles$drugbank_id)
# Split targets and ATC codes by drugbank_id to create lists
drug_target_list <- split(drug_target$gene_name,drug_target$drugbank_id)
drug_target_list <- drug_target_list[drug_id]
drug_atc_list <- split(drug_atc$atc_code,drug_atc$drugbank_id)
drug_atc_list  <- drug_atc_list[drug_id]
# Map DrugCombDB drug names to DrugBank IDs
drugcombdb_ID <- read.csv("Data/drugcombdb_drug to db.csv",check.names=F)
drugcombdb_ID <- drugcombdb_ID[,c(1,6)]
drugcombdb_ID <- drugcombdb_ID [which(drugcombdb_ID[,2]!=""),]
# Read drug combination synergy scores (Loewe additivity)
drugcombdb <- read.csv("Data/drugcombs_scored.csv")
drugcombdb <- drugcombdb[,c(2,3,4,7)]
drugcombdb <- merge(drugcombdb,drugcombdb_ID,by.x="Drug1",by.y="drug")
drugcombdb <- merge(drugcombdb,drugcombdb_ID,by.x="Drug2",by.y="drug")
drugcombdb <- drugcombdb[,c(5,6,3,4)]
colnames(drugcombdb) <- c("Drug1","Drug2","Cell_line","Loewe")
# Normalize drug pairs: sort alphabetically so that (A,B) == (B,A)
drugcombdb <- apply(drugcombdb,1,function(x){
  x[c(1,2)] <- sort(as.character(x[c(1,2)]))
  x
})
drugcombdb <- as.data.frame(t(drugcombdb))
drugcombdb$Loewe <- as.numeric(drugcombdb$Loewe)
# Read cell line metadata with Cosmic IDs
drugcombdb_cell_line <- read.csv("Data/cell_Line.csv")
drugcombdb_cell_line <- drugcombdb_cell_line[which(!is.na(drugcombdb_cell_line$cosmicId)),]
# Read landmark gene list used in L1000 project
landmaker_gene <- read.table("Data/landmaker_info.txt")
# Read gene expression data
cell_exp <- read.csv("Data/OmicsExpressionProteinCodingGenesTPMLogp1.csv",row.names = 1,check.names = F)
gene_name <- unlist(lapply(strsplit(colnames(cell_exp), ' \\('),function(x) x[1]))
colnames(cell_exp) <- gene_name
cell_exp <- scale(cell_exp)
gene <- intersect(landmaker_gene$V1,colnames(cell_exp))
cell_exp <- cell_exp[,gene]
# Read additional cell line information
cell_info <- read.csv("Data/Model.csv",check.names = F)
# Filter drugs to only those shared between previous datasets and drug combinations
drugcombdb_drug_id <- intersect(drug_id,unique(c(drugcombdb$Drug1,drugcombdb$Drug2)))
drugcombdb <- drugcombdb[which(drugcombdb$Drug1%in%drugcombdb_drug_id&drugcombdb$Drug2%in%drugcombdb_drug_id),]
drugcombdb_drug_id <- unique(c(drugcombdb$Drug1,drugcombdb$Drug2))
drugcombdb_drug_id <- sort(drugcombdb_drug_id)
# Manual correction of mismatched cell line names
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="MDA-MB-231/ATCC"),1] <- "MDA-MB-231"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="A549/ATCC"),1] <- "A549"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="NCI/ADR-RES"),1] <- "NCI\\\\/ADR-RES"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="UWB1289"),1] <- "UWB1289+BRCA1"
# Match cell lines using COSMIC ID to link different naming systems
cell_info_cosmicid <- cell_info[,c(1,3,42)]
cell_info_cosmicid <- cell_info_cosmicid[which(!is.na(cell_info_cosmicid$COSMICID)),]
cell_info_cosmicid <- merge(drugcombdb_cell_line,cell_info_cosmicid,by.x = "cosmicId",by.y = "COSMICID")
cell_info_cosmicid <- cell_info_cosmicid[,c(2,4)]
drugcombdb1 <- merge(drugcombdb,cell_info_cosmicid,by.x = "Cell_line",by.y = "cellName")
drugcombdb1 <- drugcombdb1[,c(2,3,5,4)]
# Fix other mismatches directly in drugcombdb
drugcombdb[which(drugcombdb$Cell_line=="DLD1"),3] <- "DLD-1"
drugcombdb[which(drugcombdb$Cell_line=="ZR751"),3] <- "ZR-75-1"
drugcombdb[which(drugcombdb$Cell_line=="KPL1"),3] <- "KPL-1"
drugcombdb[which(drugcombdb$Cell_line=="COLO320DM"),3] <- "COLO-320"
drugcombdb[which(drugcombdb$Cell_line=="EW-8"),3] <- "EW-8"
drugcombdb[which(drugcombdb$Cell_line=="Huh-7"),3] <- "HuH-7"
cell_info_cellname <- cell_info[which(cell_info$CellLineName%in%c("L-428","L-1236","HDLM-2","HuH-7","TC-32","RD","SMS-CTR",
                                                                  "DLD-1","ZR-75-1","KPL-1","COLO-320","TC-71","U-HO1","Rh36","EW8")),c(1,3)]
drugcombdb2 <- merge(drugcombdb,cell_info_cellname,by.x = "Cell_line", by.y = "CellLineName")
drugcombdb2 <- drugcombdb2[,c(2,3,5,4)]
drugcombdb <- rbind(drugcombdb1,drugcombdb2)
colnames(drugcombdb) <- c("Drug1","Drug2","Cell_line","Loewe")
cell_id <- sort(intersect(rownames(cell_exp),unique(drugcombdb$Cell_line)))
drugcombdb <- drugcombdb[which(drugcombdb$Cell_line%in%cell_id),]
train_cell_exp <- cell_exp[cell_id,]


Gosets_BP <- read.csv("Data/Gosets_BP.csv")
GO_similarity_matrix <- matrix(0, nrow = length(drug_target_list), ncol = length(drug_target_list), 
                               dimnames = list(names(drug_target_list), names(drug_target_list)))
# Precompute Jaccard index between each drug's targets and each GO term
GO_term_list <- lapply(strsplit(as.character(Gosets_BP$genes), ","), function(x) trimws(x))
names(GO_term_list) <- Gosets_BP$pathway
precompute_jaccard <- function(drug_targets, GO_terms) {
  jaccard_matrix <- matrix(0, nrow = length(drug_targets), ncol = length(GO_terms),dimnames = list(names(drug_targets),names(GO_terms)))
  total <- length(drug_targets) * length(GO_terms)
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  counter <- 0
  
  for (i in seq_along(drug_targets)) {
    for (j in seq_along(GO_terms)) {
      jaccard_matrix[i, j] <- length(intersect(drug_targets[[i]], GO_terms[[j]])) / length(union(drug_targets[[i]], GO_terms[[j]]))
      counter <- counter + 1
      setTxtProgressBar(pb, counter)
    }
  }

  close(pb)
  return(jaccard_matrix)
}
# Compute final GO-based drug similarity: sum product of Jaccard scores across GO terms
jaccard_matrix <- precompute_jaccard(drug_target_list, GO_term_list)
for(i in 1:(length(drug_target_list) - 1)){
  drug1_jaccard <- jaccard_matrix[i, ]
  
  for (j in (i+1):length(drug_target_list)) {
    drug2_jaccard <- jaccard_matrix[j, ]
    similarity <- sum(drug1_jaccard * drug2_jaccard, na.rm = TRUE)
    
    if (!is.na(similarity)) {
      GO_similarity_matrix[i, j] <- similarity
      GO_similarity_matrix[j, i] <- similarity
    }
  }
  print(i)
}

# Load required libraries
library(dplyr)    # For data manipulation
library(tidyr)    # For reshaping data
# Read drug-target interactions (human only, species ID 9606)
drug_target <- read.csv("Data/drug_target_9606.csv")
drug_target <- drug_target[which(drug_target$gene_name != ""), ]  # Remove entries with empty gene names
drug_target <- drug_target[, c(1, 7)]  # Keep only drugbank_id and gene_name
drug_target <- drug_target[!duplicated(drug_target), ]  # Remove duplicate rows
drug_target <- drug_target[order(drug_target$drugbank_id), ]  # Sort by drugbank_id
# Read drug-ATC code mappings
drug_atc <- read.csv("Data/drug_atc.csv")
drug_atc <- drug_atc[which(drug_atc$atc_code != ""), ]  # Remove entries with empty ATC codes
# Read SMILES strings for drugs using data.table::fread (fast reading)
drugbank_smiles <- data.table::fread("Data/drug_smiles.csv")
drugbank_smiles <- drugbank_smiles[which(drugbank_smiles$smiles != ''), ]
# Find common drug IDs present in all three datasets: target, ATC, and SMILES
drug_id <- intersect(
  intersect(unique(drug_target$drugbank_id), unique(drug_atc$drugbank_id)),
  drugbank_smiles$drugbank_id
)
# Split targets and ATC codes by drugbank_id to create lists
drug_target_list <- split(drug_target$gene_name, drug_target$drugbank_id)
drug_target_list <- drug_target_list[drug_id]  # Keep only common drugs
drug_atc_list <- split(drug_atc$atc_code, drug_atc$drugbank_id)
drug_atc_list <- drug_atc_list[drug_id]
# Map DrugCombDB drug names to DrugBank IDs
drugcombdb_ID <- read.csv("Data/drugcombdb_drug to db.csv", check.names = F)
drugcombdb_ID <- drugcombdb_ID[, c(1, 6)]  # Select columns: drug name and drugbank_id
drugcombdb_ID <- drugcombdb_ID[which(drugcombdb_ID[, 2] != ""), ]
# Read drug combination synergy scores (Loewe additivity)
drugcombdb <- read.csv("Data/drugcombs_scored.csv")
drugcombdb <- drugcombdb[, c(2, 3, 4, 7)]  # Keep Drug1, Drug2, Cell_line, Loewe score
# Merge to get DrugBank IDs for both drugs in each pair
drugcombdb <- merge(drugcombdb, drugcombdb_ID, by.x = "Drug1", by.y = "drug")
drugcombdb <- merge(drugcombdb, drugcombdb_ID, by.x = "Drug2", by.y = "drug")
drugcombdb <- drugcombdb[, c(5, 6, 3, 4)]
colnames(drugcombdb) <- c("Drug1", "Drug2", "Cell_line", "Loewe")
# Normalize drug pairs: sort alphabetically so that (A,B) == (B,A)
drugcombdb <- apply(drugcombdb, 1, function(x) {
  x[c(1, 2)] <- sort(as.character(x[c(1, 2)]))
  x
})
drugcombdb <- as.data.frame(t(drugcombdb))  # Transpose back after apply
drugcombdb$Loewe <- as.numeric(drugcombdb$Loewe)
# Read cell line metadata with Cosmic IDs
drugcombdb_cell_line <- read.csv("Data/cell_Line.csv")
drugcombdb_cell_line <- drugcombdb_cell_line[which(!is.na(drugcombdb_cell_line$cosmicId)), ]
# Read landmark gene list used in L1000 project
landmaker_gene <- read.table("Data/L1000/landmaker_info.txt")
# Read gene expression data (TPM + log transformed)
cell_exp <- read.csv("Data/CellLine_Gene_exp/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                     row.names = 1, check.names = F)
# Extract clean gene names from messy column headers (e.g., "GENE_NAME (ENSG...)" -> "GENE_NAME")
gene_name <- unlist(lapply(strsplit(colnames(cell_exp), ' \\('), function(x) x[1]))
colnames(cell_exp) <- gene_name
# Normalize expression values (Z-score scaling)
cell_exp <- scale(cell_exp)
# Intersect genes with L1000 landmark genes
gene <- intersect(landmaker_gene$V1, colnames(cell_exp))
cell_exp <- cell_exp[, gene]  # Keep only landmark genes
# Read additional cell line information
cell_info <- read.csv("Data/CellLine_Gene_exp/Model.csv", check.names = F)
# Filter drugs to only those shared between previous datasets and drug combinations
drugcombdb_drug_id <- intersect(drug_id, unique(c(drugcombdb$Drug1, drugcombdb$Drug2)))
drugcombdb <- drugcombdb[which(drugcombdb$Drug1 %in% drugcombdb_drug_id & 
                                 drugcombdb$Drug2 %in% drugcombdb_drug_id), ]
drugcombdb_drug_id <- unique(c(drugcombdb$Drug1, drugcombdb$Drug2))
drugcombdb_drug_id <- sort(drugcombdb_drug_id)
# Manual correction of mismatched cell line names
setdiff(unique(drugcombdb$Cell_line), drugcombdb_cell_line$cellName)  # Identify mismatches
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="MDA-MB-231/ATCC"), 1] <- "MDA-MB-231"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="A549/ATCC"), 1] <- "A549"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="NCI/ADR-RES"),1] <- "NCI\\\\/ADR-RES"
drugcombdb_cell_line[which(drugcombdb_cell_line$cellName=="UWB1289"),1] <- "UWB1289+BRCA1"
# Match cell lines using COSMIC ID to link different naming systems
cell_info_cosmicid <- cell_info[, c(1, 3, 42)]  # Extract sample_id, CellLineName, COSMICID
cell_info_cosmicid <- cell_info_cosmicid[which(!is.na(cell_info_cosmicid$COSMICID)), ]
cell_info_cosmicid <- merge(drugcombdb_cell_line, cell_info_cosmicid, by.x = "cosmicId", by.y = "COSMICID")
cell_info_cosmicid <- cell_info_cosmicid[, c(2, 4)]
drugcombdb1 <- merge(drugcombdb, cell_info_cosmicid, by.x = "Cell_line", by.y = "cellName")
drugcombdb1 <- drugcombdb1[, c(2, 3, 5, 4)]
# Fix other mismatches directly in drugcombdb
drugcombdb[which(drugcombdb$Cell_line == "DLD1"), 3] <- "DLD-1"
drugcombdb[which(drugcombdb$Cell_line == "ZR751"), 3] <- "ZR-75-1"
drugcombdb[which(drugcombdb$Cell_line == "KPL1"), 3] <- "KPL-1"
drugcombdb[which(drugcombdb$Cell_line == "COLO320DM"), 3] <- "COLO-320"
drugcombdb[which(drugcombdb$Cell_line == "EW-8"), 3] <- "EW-8"
drugcombdb[which(drugcombdb$Cell_line == "Huh-7"), 3] <- "HuH-7"
# Get remaining cells from direct name mapping
cell_info_cellname <- cell_info[which(cell_info$CellLineName %in% c(
  "L-428","L-1236","HDLM-2","HuH-7","TC-32","RD","SMS-CTR",
  "DLD-1","ZR-75-1","KPL-1","COLO-320","TC-71","U-HO1","Rh36","EW8"
)), c(1,3)]
drugcombdb2 <- merge(drugcombdb, cell_info_cellname, by.x = "Cell_line", by.y = "CellLineName")
drugcombdb2 <- drugcombdb2[, c(2, 3, 5, 4)]
# Combine both matched subsets
drugcombdb <- rbind(drugcombdb1, drugcombdb2)
colnames(drugcombdb) <- c("Drug1", "Drug2", "Cell_line", "Loewe")
# Final filtering: keep only cell lines with gene expression data available
cell_id <- sort(intersect(rownames(cell_exp), unique(drugcombdb$Cell_line)))
drugcombdb <- drugcombdb[which(drugcombdb$Cell_line %in% cell_id), ]
train_cell_exp <- cell_exp[cell_id, ]  # Subset expression matrix
# Load GO Biological Process (BP) gene sets
Gosets_BP <- read.csv("Data/Gosets_BP.csv")
# Initialize empty similarity matrices between drugs
GO_similarity_matrix <- matrix(0, nrow = length(drug_target_list), ncol = length(drug_target_list),
                               dimnames = list(names(drug_target_list), names(drug_target_list)))
# Parse GO terms: split gene lists and clean whitespace
GO_term_list <- lapply(strsplit(as.character(Gosets_BP$genes), ","), function(x) trimws(x))
names(GO_term_list) <- Gosets_BP$pathway
# Precompute Jaccard index between each drug's targets and each GO term
precompute_jaccard <- function(drug_targets, GO_terms) {
  jaccard_matrix <- matrix(0, nrow = length(drug_targets), ncol = length(GO_terms),
                           dimnames = list(names(drug_targets), names(GO_terms)))
  total <- length(drug_targets) * length(GO_terms)
  pb <- txtProgressBar(min = 0, max = total, style = 3)
  counter <- 0
  
  for (i in seq_along(drug_targets)) {
    for (j in seq_along(GO_terms)) {
      intersection <- length(intersect(drug_targets[[i]], GO_terms[[j]]))
      union <- length(union(drug_targets[[i]], GO_terms[[j]]))
      jaccard_matrix[i, j] <- ifelse(union == 0, 0, intersection / union)
      counter <- counter + 1
      setTxtProgressBar(pb, counter)
    }
  }
  close(pb)
  return(jaccard_matrix)
}
jaccard_matrix <- precompute_jaccard(drug_target_list, GO_term_list)
# Compute final GO-based drug similarity: sum product of Jaccard scores across GO terms
for(i in 1:(length(drug_target_list) - 1)){
  drug1_jaccard <- jaccard_matrix[i, ]
  
  for (j in (i+1):length(drug_target_list)) {
    drug2_jaccard <- jaccard_matrix[j, ]
    similarity <- sum(drug1_jaccard * drug2_jaccard, na.rm = TRUE)
    
    if (!is.na(similarity)) {
      GO_similarity_matrix[i, j] <- similarity
      GO_similarity_matrix[j, i] <- similarity
    }
  }
  print(i)  # Progress indicator
}
# Compute ATC-based drug similarity
ATC_similarity_matrix <- matrix(0, nrow = length(drug_atc_list), ncol = length(drug_atc_list), 
                                dimnames = list(names(drug_atc_list), names(drug_atc_list)))
for(i in 1:(length(drug_atc_list)-1)){
  for(j in (i+1):length(drug_atc_list)){
    atc_list1 <- drug_atc_list[[i]]
    atc_list2 <- drug_atc_list[[j]]
    atc_comb <- expand.grid("atc_list1"= atc_list1,"atc_list2" = atc_list2,stringsAsFactors=F)
    atc <- c()
    for(k in 1:nrow(atc_comb)){
      atc1 <- length(intersect(substr(atc_comb[k,1],1,1),substr(atc_comb[k,2],1,1)))/length(union(substr(atc_comb[k,1],1,1),substr(atc_comb[k,2],1,1)))
      atc2 <- length(intersect(substr(atc_comb[k,1],2,3),substr(atc_comb[k,2],2,3)))/length(union(substr(atc_comb[k,1],2,3),substr(atc_comb[k,2],2,3)))
      atc3 <- length(intersect(substr(atc_comb[k,1],4,4),substr(atc_comb[k,2],4,4)))/length(union(substr(atc_comb[k,1],4,4),substr(atc_comb[k,2],4,4)))
      atc4 <- length(intersect(substr(atc_comb[k,1],5,5),substr(atc_comb[k,2],5,5)))/length(union(substr(atc_comb[k,1],5,5),substr(atc_comb[k,2],5,5)))
      atc5 <- length(intersect(substr(atc_comb[k,1],6,7),substr(atc_comb[k,2],6,7)))/length(union(substr(atc_comb[k,1],6,7),substr(atc_comb[k,2],6,7)))
      atc <- c(atc,mean(c(atc1,atc2,atc3,atc4,atc5)))
    }
    SATC <- mean(atc)
    ATC_similarity_matrix[i,j] <- SATC
    ATC_similarity_matrix[j,i] <- SATC
  }
  print(i)
}
