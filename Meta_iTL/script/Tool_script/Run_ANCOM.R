
deps = c("exactRankTests", "nlme", "dplyr", "ggplot2", "compositions")
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(dep)
  }
  library(dep, character.only = TRUE)
}

#args[4] will contain path for the ancom code


args <- commandArgs(trailingOnly = TRUE)

if (length(args) <= 3) {
  print(args)
  print(length(args))
  stop("At least three arguments must be supplied", call.=FALSE)
}

source(args[[4]])


feature_table <- read.csv(args[1], row.names = 1, check.names = FALSE)
meta <- read.csv(args[2],row.names = 1, check.names = FALSE)


groupings <- as.data.frame(meta$Group)
colnames(groupings) <- "Group"
rownames(groupings) <- meta$Sample_ID

#number of samples
sample_num <- length(colnames(feature_table))
grouping_num <- length(rownames(groupings))

if(sample_num != grouping_num){
  message("The number of samples in the feature table and the groupings table are unequal")
  message("Will remove any samples that are not found in either the feature table or the groupings table")
}

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

groupings$Sample <- rownames(groupings)

prepro <- feature_table_pre_process(feature_table = feature_table, meta_data = groupings, sample_var = 'Sample', 
                                    group_var = NULL, out_cut = 0.05, zero_cut = 0.90,
                                    lib_cut =0.000001, neg_lb=FALSE)

feature_table <- prepro$feature_table
metadata <- prepro$meta_data
struc_zero <- prepro$structure_zeros

#run ancom
main_var <- colnames(groupings)[1]
p_adj_method = "BH"
alpha=0.05
adj_formula=NULL
rand_formula=NULL
res <- ANCOM(feature_table = feature_table, meta_data = metadata, struc_zero = struc_zero, main_var = main_var, p_adj_method = p_adj_method,
             alpha=alpha, adj_formula = adj_formula, rand_formula = rand_formula)


write.table(res$out, file=args[3], quote=FALSE, sep="\t", col.names = NA)
