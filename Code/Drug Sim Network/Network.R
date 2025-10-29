drug_target <- read.csv("Data/drug_target_9606.csv")
drug_target <- drug_target[which(drug_target$gene_name!=""),]
drug_target <- drug_target[,c(1,7)]
drug_target <- drug_target[!duplicated(drug_target),]
drug_target <- drug_target[order(drug_target$drugbank_id),]

drug_atc <- read.csv("Data/drug_atc.csv")
drug_atc <- drug_atc[which(drug_atc$atc_code!=""),]

drugbank_smiles <- data.table::fread("Data/drug_smiles.csv")
drugbank_smiles <- drugbank_smiles[which(drugbank_smiles$smiles!=''),]

drug_id <- intersect(intersect(unique(drug_target$drugbank_id),unique(drug_atc$drugbank_id)),drugbank_smiles$drugbank_id)


drug_target_list <- split(drug_target$gene_name,drug_target$drugbank_id)
drug_target_list <- drug_target_list[drug_id]

drug_atc_list <- split(drug_atc$atc_code,drug_atc$drugbank_id)
drug_atc_list  <- drug_atc_list[drug_id]


GO_similarity_matrix <- matrix(0, nrow = length(drug_target_list), ncol = length(drug_target_list), 
                               dimnames = list(names(drug_target_list), names(drug_target_list)))

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
