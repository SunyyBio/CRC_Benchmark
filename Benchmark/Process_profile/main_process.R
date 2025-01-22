
library(dplyr)

#1.进行数据过滤
#CTR_CRC"class到SGB层级是：至少在2个研究中平均值大于0.00001，在90%的样本中不为0，以及方差不为0
#CTR_CRC"ko到uniref层级是：至少在8个研究中平均值大于0.0001，在90%的样本中不为0，以及方差不为0
#CTR_ADA"class到SGB层级是：至少在2个研究中平均值大于0.00001，在90%的样本中不为0，以及方差不为0
rm(list = ls())
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data/",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")
sink(paste(script_dir,"CRC.txt",sep=""))

source(paste(script_dir,"/1_Profile_process.R",sep=""))
for (feature_type in c("species")){
  memory.limit(size = 10000) # 8GB内存限制，您可以根据需要调整大小
  feature_file <- paste0(data_script_dir,feature_type,"/",feature_type,"_merged.csv")
  output_file<-paste0(data_script_dir,feature_type,"/",sep="")
  for (group_info in list(c('CTR', 'CRC'))){
    study_num=2
    mean_ab=0.00001
    CTR_CRC(feature_file,output_file,group_info,study_num,mean_ab)
  }
}

#获取额外的验证队列的dataframe
for (dir_ in c("class","order","family","genus","species","t_sgb")){
  group_info=c("CTR","CRC")
  
  meta.all <- read.csv(paste(data_script_dir,"/metaA_all.csv",sep=""),sep = ',')
  feature_file <- paste0(data_script_dir,feature_type,"/",feature_type,"_merged.csv")
  output_file<-paste0(data_script_dir,feature_type,"/",sep="")
  feat.ab.all<-read.csv(feature_file,sep = ',',row.names = 1)
  feat.ab.all[is.na(feat.ab.all)] <- 0
  feat.ab.all <- feat.ab.all[, colSums(feat.ab.all) != 0]
  meta.all$Sample_ID <- as.character(meta.all$Sample_ID)
  meta.all <- meta.all[meta.all$Sample_ID %in% colnames(feat.ab.all),]
  crc.studies <- c("CHN_SH-CRC-3","CHN_SH-CRC-4")
  meta.crc <- meta.all %>%
    filter(Study %in% crc.studies) %>%
    filter(Group %in% group_info )
  
  stopifnot(all(meta.crc$Sample_ID %in% colnames(feat.ab.all)))
  #训练集的物种相对丰度
  feat.ab.all=as.matrix(feat.ab.all)
  feat.rel.crc <- feat.ab.all[,meta.crc$Sample_ID]
  feat.rel.crc=prop.table(feat.rel.crc,2)
  
  fn.tax.rel.ab <- paste0(output_file,'feature_rare_ext_', paste(group_info, collapse = '_'),'.csv')
  if (!dir.exists(dirname(fn.tax.rel.ab))) {
    # 如果目录不存在，则创建目录
    dir.create(dirname(fn.tax.rel.ab), recursive = TRUE)
  }
  write.table(feat.rel.crc, file=fn.tax.rel.ab, quote=FALSE, sep='\t',
              row.names=TRUE, col.names=TRUE)

}

#2.调整批次效应
source(paste(script_dir,"/1_Profile_process_batch_corred.R",sep=""))
for (class_dir in c("class")){
  for (groups in c("CTR_CRC")){
    MMUPHin_info(class_dir,groups)
  }
}
