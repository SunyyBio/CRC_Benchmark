package_list = c("dplyr","ggplot2","randomForest","caret","A3","Boruta","ImageGP","ROCR","corrplot","vegan","Hmisc","Maaslin2","MMUPHin")

###CHECK PACKAGES
site="https://mirrors.tuna.tsinghua.edu.cn/CRAN"
for(p in package_list){
  if(!suppressWarnings(suppressMessages(require(p, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)))){
    install.packages(p, repos=site)
    suppressWarnings(suppressMessages(library(p, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)))
  }
}
rm(list=ls())
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data/",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")


raw_lefse_MaAsLin<-function(study,file_name){
  "
  Raw data analysis
  "
  meta.all <- read.csv(file = paste(data_script_dir,"/meta.csv",sep=""),sep=',', stringsAsFactors = FALSE, header = TRUE)
  meta.all <- meta.all[,c("Sample_ID","Study","Group")]
  
  feat.all <- read.csv(file =paste(data_script_dir,"/feature_rare_CTR_ADA.csv", sep=""),
                       sep='\t', stringsAsFactors = FALSE, header = TRUE)
  feat.all <- apply(feat.all + 1, 2, function(x){log(x)})
  feat.all <- as.data.frame(t(feat.all))

  row.names(meta.all) <- meta.all$Sample_ID
  sample_table <- subset(meta.all, Study %in% study)
  common_samples <- intersect(row.names(sample_table), colnames(feat.all))
  sample_table <- meta.all[common_samples, ]
  feature_table <- feat.all[,common_samples]
  sample_table$StudyID <- factor(sample_table $Study)
  
  merged_data_normalized <- feature_table - apply(feature_table, 2, min)
  scaled_range <- apply(feature_table, 2, max) - apply(feature_table, 2, min)
  feature_table <- merged_data_normalized / scaled_range

  pacman::p_load(tidyverse,microeco,magrittr)
  tax <-data.frame(kingdom=NA,
                   Phylum=NA,
                   Class=NA,
                   Order=NA,
                   Family=NA,
                   Genus=NA,
                   Species=row.names(feature_table))
  row.names(tax) <-row.names(feature_table)
  dataset <- microtable$new(sample_table = sample_table,
                            otu_table = as.data.frame(feature_table),
                            tax_table =tax )
  dir.create(paste0(feature_script_dir,"/raw/all_feature/",sep=""),recursive =TRUE)
  lefse <- trans_diff$new(dataset = dataset,
                          method = "lefse",
                          group = "Group",
                          alpha = 0.05,
                          taxa_level="species",
                          p_adjust_method = "none",
                          lefse_subgroup = NULL)
  write.table(lefse$res_diff,paste0(feature_script_dir,"/raw/all_feature/",file_name,"_lefse.csv",sep=""))
  fit_data = Maaslin2(input_data     = t(feature_table),
                      input_metadata = sample_table,
                      min_prevalence = 0,
                      normalization  = "NONE",
                      output         = paste0(feature_script_dir,"/raw/all_feature/",file_name,"_MaAslin",sep=""),
                      fixed_effects  = c("Group"),
                     reference = c("Group,CTR"))
  write.csv(sample_table,paste0(data_script_dir,"/meta_CTR_ADA.csv",sep=""),sep="\t")
  write.csv(feature_table,paste0(data_script_dir,"/feat_CTR_ADA.csv",sep=""),sep="\t")
  
  feature_table_Path=paste(data_script_dir,"/feat_CTR_ADA.csv",sep="")
  Groupings_Path=paste(data_script_dir,"/meta_CTR_ADA.csv",sep="")
  
  TOOL_DIR <-paste(script_dir,"/Tool_script/",sep="")
  ANCOM_DIR<-paste(script_dir,"/Tool_script/Ancom2_Script",sep="")

  ANCOM_Script_Path <- file.path(ANCOM_DIR, "ancom_v2.1.R")
  Run_ANCOM_Script_Path <- file.path(TOOL_DIR, "Run_ANCOM.R")
  
  out_file_ancom <-paste0(feature_script_dir,"/raw/all_feature/",file_name,"_Ancom.csv",sep="")
  cmd <- sprintf("Rscript \"%s\" \"%s\" \"%s\" \"%s\" \"%s\"", Run_ANCOM_Script_Path, feature_table_Path, Groupings_Path, out_file_ancom, ANCOM_Script_Path)
  cmd_time <- system.time({
    system(cmd)
  })
}

#Filter features in each study in the source domain
study<-c('US-CRC-2','AT-CRC','JPN-CRC','FR-CRC','CHN_WF-CRC','ITA-CRC',"CHN_SH-CRC-2","US-CRC-3","CHN_SH-CRC-4")
for (studys in study){
  file_name=studys
  raw_lefse_MaAsLin(study,file_name)
}

#Filter features in all queues in the source domain
study<-c("AT-CRC","FR-CRC","ITA-CRC","JPN-CRC", "US-CRC-2","US-CRC-3","CHN_WF-CRC","CHN_SH-CRC-2")
file_name<-"FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
raw_lefse_MaAsLin(study,file_name)

#Filter features in all queues in the source domain
study<-c("AT-CRC","FR-CRC","ITA-CRC","JPN-CRC", "US-CRC-2","US-CRC-3","CHN_SH-CRC-4","CHN_SH-CRC-2")
file_name<-"FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
raw_lefse_MaAsLin(study,file_name)
#####################################################################
#Features of the select samples from target domain dataset filtered out
#####################################################################
rm(list=ls())
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data/",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")

###########################################################################################
#Feature selection on Meta-iTL dataset
##########################################################################################
Meta_iTL_all_feature<-function(source_study,target_study,ratio){
  feature_table <- read.csv(file =paste0(data_script_dir,source_study,"_",target_study,'/filtered_',source_study,"_",target_study,"_",ratio,'_train.csv',sep=""),sep=',',stringsAsFactors = FALSE, row.names =1, header = TRUE,check.name = FALSE)
  meta<-feature_table[c("Study","Group")]
  feature_table <-t(feature_table[,3:ncol(feature_table)])
  
  merged_data_normalized <- feature_table - apply(feature_table, 2, min)
  scaled_range <- apply(feature_table, 2, max) - apply(feature_table, 2, min)
  feature_table <- merged_data_normalized / scaled_range
  
  pacman::p_load(tidyverse,microeco,magrittr)
  tax <-data.frame(kingdom=NA,
                   Phylum=NA,
                   Class=NA,
                   Order=NA,
                   Family=NA,
                   Genus=NA,
                   Species=row.names(feature_table))
  row.names(tax) <-row.names(feature_table)
  dataset <- microtable$new(sample_table =as.data.frame(meta),
                            otu_table = as.data.frame(feature_table),
                            tax_table =tax )
  out_put_dir<-paste0(feature_script_dir,"/Meta_iTL/",source_study,"_",target_study,'/',ratio,"/",sep="")
  dir.create(out_put_dir,recursive =TRUE)
  lefse <- trans_diff$new(dataset = dataset,
                          method = "lefse",
                          group = "Group",
                          alpha =0.05,
                          taxa_level="species",
                          p_adjust_method = "none",
                          lefse_subgroup = NULL)
  write.table(lefse$res_diff,paste(out_put_dir,'Meta_iTL_all_cohort_lefse.csv',sep=""))

  fit_data = Maaslin2(input_data     = t(feature_table),
                      input_metadata = meta,
                      min_prevalence = 0,
                      normalization  = "NONE",
                      output         = paste(out_put_dir,'Meta_iTL_all_cohort_MaAslin/',sep=""),
                      fixed_effects  = c("Group"),
                      random_effects = c('Study'),
                      reference = c("Group,CTR"))

  meta$Sample_ID <- colnames(feature_table)
  write.csv(meta,paste(out_put_dir,'/meta_CTR_ADA.csv',sep=""),sep="\t",row.names=TRUE)
  write.csv(feature_table,paste(out_put_dir,"/feat_CTR_ADA.csv",sep=""),sep="\t")
  
  ASV_table_Path=paste(out_put_dir,'/feat_CTR_ADA.csv',sep="")
  Groupings_Path=paste(out_put_dir,"/meta_CTR_ADA.csv",sep="")
  
  TOOL_DIR <-paste(script_dir,"/Tool_script/",sep="")
  ANCOM_DIR<-paste(script_dir,"/Tool_script/Ancom2_Script",sep="")
  ANCOM_Script_Path <- file.path(ANCOM_DIR, "ancom_v2.1.R")
  Run_ANCOM_Script_Path <- file.path(TOOL_DIR, "Run_ANCOM.R")

  out_file_ancom <- paste(out_put_dir,"Meta_iTL_all_cohort_Ancom.csv",sep="")
  cmd <- sprintf("Rscript \"%s\" \"%s\" \"%s\" \"%s\" \"%s\"", Run_ANCOM_Script_Path, ASV_table_Path, Groupings_Path, out_file_ancom, ANCOM_Script_Path)
  cmd_time <- system.time({
    system(cmd)
  })
  print("MNN_all_studys-Succeed==============================================")
}
##########################################################################################
#Each cohort in the Meta-iTL dataset was subjected to separate feature selection
##########################################################################################
Meta_iTL_single_feature <-function(source_study,target_study,all_studys,ratio){
  for (study in all_studys){
    feature_table <- read.csv(file =paste0(data_script_dir,source_study,"_",target_study,'/filtered_',source_study,"_",target_study,"_",ratio,'_train.csv',sep=""),sep=',',stringsAsFactors = FALSE, row.names =1, header = TRUE,check.name = FALSE)
    feature_table <- subset(feature_table, Study %in% study)
    meta<-feature_table[c("Study","Group")]
    feature_table <-t(feature_table[,3:ncol(feature_table)])
    
    merged_data_normalized <- feature_table - apply(feature_table, 2, min)
    scaled_range <- apply(feature_table, 2, max) - apply(feature_table, 2, min)
    feature_table <- merged_data_normalized / scaled_range

    tryCatch({
      pacman::p_load(tidyverse,microeco,magrittr)
      tax <-data.frame(kingdom=NA,
                       Phylum=NA,
                       Class=NA,
                       Order=NA,
                       Family=NA,
                       Genus=NA,
                       Species=row.names(feature_table))
      row.names(tax) <-row.names(feature_table)
      dataset <- microtable$new(sample_table =as.data.frame(meta),
                                otu_table = as.data.frame(feature_table),
                                tax_table =tax )

      lefse <- trans_diff$new(dataset = dataset,
                              method = "lefse",
                              group = "Group",
                              alpha =0.05,
                              taxa_level="species",
                              p_adjust_method = "none",
                              lefse_subgroup = NULL)

    out_put_dir<-paste0(feature_script_dir,"/Meta_iTL/",source_study,"_",target_study,'/',ratio,"/",sep="")
      dir.create(out_put_dir,recursive =TRUE)
      write.table(lefse$res_diff,paste(out_put_dir,study,"_lefse.csv",sep=""))
    }, error = function(e) {
      dir.create(out_put_dir,recursive =TRUE)
      dir.create(paste(out_put_dir,study,"_MaAslin/",sep=""),recursive =TRUE)

      column_names <- c("Comparison","Taxa","Method","Group","LDA","P.unadj","P.adj","Significance")
      df <- data.frame(matrix(ncol = length(column_names), nrow = 0))
      colnames(df) <- column_names
      write.table(df,paste(out_put_dir,study,"_lefse.csv",sep=""))
    })
    tryCatch({
      fit_data = Maaslin2(input_data     = t(feature_table),
                          input_metadata = meta,
                          min_prevalence = 0,
                          normalization  = "NONE",
                          output         = paste(out_put_dir,study,"_MaAslin",sep=""),
                          fixed_effects  = c("Group"),
                          reference = c("Group,CTR"))
    }, error = function(e) {
      column_names1 <- c('feature', 'metadata', 'value', 'coef', 'stderr', 'N', 'N.not.0', 'pval', 'qval')
      df1 <- data.frame(matrix(ncol = length(column_names1), nrow = 0))
      colnames(df1) <- column_names1
      write.table(df1,paste(out_put_dir,study,"_MaAslin/all_results.tsv",sep=""),sep='\t')
    })

    tryCatch({
      meta$Sample_ID <- colnames(feature_table)
      write.csv(meta,paste(out_put_dir,"/meta_CTR_ADA.csv",sep=""),sep="\t",row.names=TRUE)
      write.csv(feature_table,paste(out_put_dir,"/feat_CTR_ADA.csv",sep=""),sep="\t")
      
      ASV_table_Path=paste(out_put_dir,"/feat_CTR_ADA.csv",sep="")
      Groupings_Path=paste(out_put_dir,"/meta_CTR_ADA.csv",sep="")
      
      TOOL_DIR <-paste(script_dir,"/Tool_script/",sep="")
      ANCOM_DIR<-paste(script_dir,"/Tool_script/Ancom2_Script",sep="")
      ANCOM_Script_Path <- file.path(ANCOM_DIR, "ancom_v2.1.R")
      Run_ANCOM_Script_Path <- file.path(TOOL_DIR, "Run_ANCOM.R")

      out_file_ancom <- paste(out_put_dir,study,"_Ancom.csv",sep="")
      cmd <- sprintf("Rscript \"%s\" \"%s\" \"%s\" \"%s\" \"%s\"", Run_ANCOM_Script_Path, ASV_table_Path, Groupings_Path, out_file_ancom, ANCOM_Script_Path)
      return_code <- system(cmd)
      if (return_code != 0) {
        stop("Command failed with return code: ", return_code)
      }
      print("Meat_iTL_single_featureSucceed==============================================")
    }, error = function(e) {
      column_names1 <- c('taxa_id', 'W', 'detected_0.9', 'detected_0.8', 'detected_0.7', 'detected_0.6')
      # 创建空DataFrame
      df2 <- data.frame(matrix(ncol = length(column_names1), nrow = 0))
      colnames(df2) <- column_names1
      write.table(df2,paste(out_put_dir,study,"_Ancom.csv",sep=""),sep='\t')
    })
  }
}

TL_main_CHN_SH<-function(){
  source_study<-"FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_CHN_WF-CRC_CHN_SH-CRC-2_US-CRC-3"
  target_study <-"CHN_SH-CRC-4"
  for (ratio in c("S0.2","S0.3","S0.4","S0.5","S0.6")){
    all_studys <-c("FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","CHN_WF-CRC","CHN_SH-CRC-2","US-CRC-3","CHN_SH-CRC-4")
    Meta_iTL_all_feature(source_study,target_study,ratio)
    Meta_iTL_single_feature(source_study,target_study,all_studys,ratio)
    print(ratio)
  }
}
TL_main_CHN_WF<-function(){
  source_study<-"FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_US-CRC-2_US-CRC-3_CHN_SH-CRC-4_CHN_SH-CRC-2"
  target_study <-"CHN_WF-CRC"
  for (ratio in c("S0.2","S0.3","S0.4","S0.5","S0.6")){
    all_studys <-c("FR-CRC","AT-CRC","ITA-CRC","JPN-CRC","US-CRC-2","CHN_WF-CRC","CHN_SH-CRC-2","US-CRC-3","CHN_SH-CRC-4")
    Meta_iTL_all_feature(source_study,target_study,ratio)
    Meta_iTL_single_feature(source_study,target_study,all_studys,ratio)
  }
}
TL_main_CHN_SH()
TL_main_CHN_WF()