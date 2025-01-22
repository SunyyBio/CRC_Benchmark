args <- commandArgs(trailingOnly = TRUE)

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


colnames(groupings)
colnames(groupings)[1] <- "places"

#apply wilcox test to rarified table
pvals <- apply(feature_table, 1, function(x) wilcox.test(x ~ groupings[,1], exact=F)$p.value)

write.table(pvals, file=args[[3]], sep="\t", col.names = NA, quote=F)


