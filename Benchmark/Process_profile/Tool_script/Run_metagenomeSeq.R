deps = c("metagenomeSeq")
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager")
    
    BiocManager::install(deps)
  }
  library(dep, character.only = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
#test if there is an argument supply
if (length(args) <= 2) {
  stop("At least three arguments must be supplied", call.=FALSE)
}


feature_table <- read.csv(args[1], row.names = 1, check.names = FALSE)
meta <- read.csv(args[2],row.names = 1, check.names = FALSE)
groupings <- as.data.frame(meta$Group)
colnames(groupings) <- "Group"
rownames(groupings) <- meta$Sample_ID
#number of samples
sample_num <- length(colnames(feature_table))
grouping_num <- length(rownames(groupings))

#check if the same number of samples are being input.
if(sample_num != grouping_num){
  message("The number of samples in the feature table and the groupings table are unequal")
  message("Will remove any samples that are not found in either the feature table or the groupings table")
}

#check if order of samples match up.
if(identical(colnames(feature_table), rownames(groupings))==T){
  message("Groupings and feature table are in the same order")
}else{
  rows_to_keep <- intersect(colnames(feature_table), rownames(groupings))
  groupings <- groupings[rows_to_keep,,drop=F]
  feature_table <- feature_table[,rows_to_keep]
  if(identical(colnames(feature_table), rownames(groupings))==T){
    message("Groupings table was re-arrange to be in the same order as the feature table")
    message("A total of ", sample_num-length(colnames(feature_table)), " from the feature_table")
    message("A total of ", grouping_num-length(rownames(groupings)), " from the groupings table")
  }else{
    stop("Unable to match samples between the feature table and groupings table")
  }
}

data_list <- list()
data_list[["counts"]] <- feature_table
data_list[["taxa"]] <- rownames(feature_table)

pheno <- AnnotatedDataFrame(groupings)
pheno
counts <- AnnotatedDataFrame(feature_table)
feature_data <- data.frame("feature"=rownames(feature_table),
                           "feature2"=rownames(feature_table))
feature_data <- AnnotatedDataFrame(feature_data)
rownames(feature_data) <- feature_data@data$feature


test_obj <- newMRexperiment(counts = data_list$counts, phenoData = pheno, featureData = feature_data)

p <- cumNormStat(test_obj, pFlag = T)
p

test_obj_norm <- cumNorm(test_obj, p=p)

fromula <- as.formula(paste(~1, colnames(groupings)[1], sep=" + "))
pd <- pData(test_obj_norm)
mod <- model.matrix(fromula, data=pd)
regres <- fitFeatureModel(test_obj_norm, mod)

res_table <- MRfulltable(regres, number = length(rownames(feature_table)))

write.table(res_table, file=args[3], quote=F, sep="\t", col.names = NA)
