package_list <- c("ggplot2", "ggridges", "cowplot", "pheatmap", "ggrepel")
site <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN"
installed_packages <- rownames(installed.packages())
for (p in package_list) {
  if (!(p %in% installed_packages)) {
    install.packages(p, repos = site)
  }
}
# Load all required packages
lapply(package_list, library, character.only = TRUE)
rm(list=ls())
options(scipen = 999)
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data/",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")
figures_script_dir <- paste0(dirname(script_dir),"/Result/figures/Fig03/",sep="")

for (analysis_level in c("species")){
  for (group in c("CTR_CRC")){
    data_type="Raw"
    tool_names <- c(ancom="ANCOM-II", lefse="LEfSe",RFECV="RFECV",maaslin2="MaAsLin2",metagenomeSeq="MetagenomeSeq", ttest="T-test",
                    wilcoxon="Wilcoxon")
    filt_results <- read.csv(paste(feature_script_dir,analysis_level,"/",data_type,"/",group,"/adj_p_",group,".csv",sep=""),row.names = 1)
    Adj_p_tabs_filt <- list()
    for(study in names(filt_results)){
      Adj_p_tabs_filt[[study]] <- filt_results[[study]]
    }
    p_sig <- 0.01
    p_sig_lefse <- 0.05
    Adj_p_all_filt <- t(do.call(rbind, Adj_p_tabs_filt))
    row.names(Adj_p_all_filt)<- row.names(filt_results)
    Adj_p_all_filt[is.na(Adj_p_all_filt)] <- 1
    for (col in colnames(Adj_p_all_filt)) {
      if (col == "lefse") {
        Adj_p_all_filt[Adj_p_all_filt[, col] < p_sig_lefse, col] <- 2
      } else {
        Adj_p_all_filt[Adj_p_all_filt[, col] < p_sig, col] <- 2
      }
    }
    Adj_p_all_filt[Adj_p_all_filt != 2] <- 0
    Adj_p_all_filt[Adj_p_all_filt==2] <- 1
    Feature_sig_count_filt <- rowSums(Adj_p_all_filt, na.rm=T)
    Feature_count_tab_filt <- Adj_p_all_filt
    Feature_count_tab_filt[is.na(Feature_count_tab_filt)] <- 0
    row_to_remove <- which(rowSums(Feature_count_tab_filt)==0)
    Feature_count_tab_filt <- Feature_count_tab_filt[-row_to_remove, ]
    
    if (!dir.exists(paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,sep=""))) {
      dir.create(paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,sep=""), recursive = TRUE)
    }
    write.csv(Feature_count_tab_filt,file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Features_found_by_each_tool_Upset_",group,".csv",sep=""))
    for(row_num in 1:length(rownames(Feature_count_tab_filt))){
      row_name <- rownames(Feature_count_tab_filt)[row_num]
      Score_for_that_row <- Feature_sig_count_filt[row_name]
      for(j in 1:ncol(Feature_count_tab_filt)){
        if(is.na(Feature_count_tab_filt[row_num, j])){
        }else if(Feature_count_tab_filt[row_num, j] == 1){
          Feature_count_tab_filt[row_num, j] <- Score_for_that_row
        }
      }
    }
    saveRDS(Feature_count_tab_filt, paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Feature_count_tab_21_09_22.RDS",sep=""))
    Feature_count_tab_filt <- readRDS(paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Feature_count_tab_21_09_22.RDS",sep=""))

    which(is.na(Feature_count_tab_filt))
    plot_info_barplot <- reshape2::melt(Feature_count_tab_filt)
    colnames(plot_info_barplot) <- c(analysis_level, "variable", "value")
    plot_info_barplot <- as.data.frame(plot_info_barplot)
    plot_info_barplot <- plot_info_barplot[, c("variable", "value")]
    
    remove_row <- which(plot_info_barplot$value==0)
    no_zero <- plot_info_barplot[-remove_row,]

    feat_sum_filt <- Feature_count_tab_filt
    feat_sum_filt[feat_sum_filt > 0] <- 1
    Tool_total_feats_found <- as.data.frame(colSums(feat_sum_filt))
    colnames(Tool_total_feats_found) <- "Total Sig Features"
    Tool_total_feats_found <- data.frame(Tool_total_feats_found)
    Tool_total_feats_found$variable <- rownames(Tool_total_feats_found)
    merged_no_zero_filt <- dplyr::inner_join(no_zero, Tool_total_feats_found,by="variable")
    intersect(no_zero$variable, Tool_total_feats_found$variable)
    
    colnames(merged_no_zero_filt)[3] <- "Total_Features"
    merged_no_zero_filt$clean_variable <- tool_names[merged_no_zero_filt$variable]
    wide_filt_tab <- reshape2::dcast(merged_no_zero_filt, variable ~ value, fun.aggregate = length)
    wide_filt_tab_RA2 <- wide_filt_tab
    wide_filt_tab_RA2[,-1] <- sweep(wide_filt_tab_RA2[,-1], 1, rowSums(wide_filt_tab_RA2[,-1]), '/')*100
    wide_filt_tab$total<-rowSums(wide_filt_tab[,-1])
    
    write.csv( wide_filt_tab,file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Overlap_feature_",group,".csv",sep=""))
    write.csv(wide_filt_tab_RA2,file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Proportion_of_repeats_for_features_in_tools_",group,".csv",sep=""))
    long_filt_melt <- reshape2::melt(wide_filt_tab_RA2)
    colnames(long_filt_melt) <- c("variable", "Score", "value")
    long_filt_melt$Method_clean <- tool_names[long_filt_melt$variable]
    
    long_filt_melt_join <- dplyr::inner_join(long_filt_melt, Tool_total_feats_found)
    colnames(long_filt_melt_join)[5] <- "Total_Hits"
    
    Filt_bars <- ggplot(long_filt_melt_join, aes(x=as.numeric(Score
    ), y=value, width=0.95, fill=Total_Hits)) +
      geom_bar(stat="identity") +
      xlab("No. tools that called feature significant") +
      ylab("") +
      ggtitle(group) +
      facet_grid(rows=vars(Method_clean), switch='y') +
      theme_classic() +
      scale_y_continuous(position = "right", breaks=c(0,30,60, 90,100), labels=c("0%","30%","60%", "90%","100%")) +
      theme(strip.text.y.left = element_text(angle=0), strip.background = element_blank()) +
      scale_x_continuous(breaks=seq(1,7), expand=c(0,0)) +
      theme(text=element_text(size=16)) +
      guides(fill=guide_legend(title="Feature Num")) +
      scale_fill_continuous(high = "#FFA061", low = "#006DA3", breaks=seq(50, 400, by=10))
    Filt_bars
    ggsave(filename=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Overlap_feature_", group,".pdf", sep=""),
           plot = Filt_bars, width = 8, height=8, units="in", dpi=100)
    write.csv(long_filt_melt_join,file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/Overlap_significant_features_",group,".csv",sep=""))
  }
}
while (!is.null(dev.list())) dev.off()

###########################################################################
#Analysis of single cohort characterization tools

rm(list=ls())
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data/",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")
figures_script_dir <- paste0(dirname(script_dir),"/Result/figures/Fig03/",sep="")

Data_set_names <- c(AT.CRC="AT-CRC",
                    FR.CRC="FR-CRC",
                    CHN_WF.CRC="CHN_WF-CRC",
                    CHN_SH.CRC="CHN_SH-CRC",
                    CHN_HK.CRC="CHN_HK-CRC",
                    US.CRC="US-CRC",
                    ITA.CRC="ITA-CRC",
                    DE.CRC="GE-CRC",
                    IND.CRC="IND-CRC",
                    JPN.CRC="JPN-CRC")
tool_names <- c(ancom="ANCOM-II", lefse="LEfSe",RFECV="RFECV",maaslin2="MaAsLin2",metagenomeSeq="metagenomeSeq", ttest="t-test",wilcoxon="Wilcoxon")
p_sig_lefse=0.05
p_sig=0.01
filt_study_tab <- list()
filt_study_tab[["nonrare"]] <- list()

for (analysis_level in c("species")){
  for (groups in c("CTR_CRC")){
    data_type="Batch"
    meta.all <- read.csv(file = paste(data_script_dir,"meta.csv",sep=""),sep=',',stringsAsFactors = FALSE, header = TRUE,row.names = 2)
    
    if (data_type=="Raw"){
      feat.all <- read.csv(file =paste(data_script_dir,analysis_level, "/feature_rare_", groups, ".csv", sep=""),
                           sep='\t', stringsAsFactors = FALSE, header = TRUE)
      feat.all <- as.data.frame(t(feat.all))
    }else if(data_type=="Raw_log"){
      feat.all <- read.csv(file =paste(data_script_dir,analysis_level, "/feature_rare_", groups, ".csv", sep=""),
                           sep='\t', stringsAsFactors = FALSE, header = TRUE)
      feat.all <- apply(feat.all + 1, 2, function(x) log(x))
      feat.all <- as.data.frame(t(feat.all))
    }else{
      feat.all <- read.csv(file=paste0(data_script_dir,analysis_level,'/',groups,'_adj_batch.csv',sep=''),row.names =1)
    }
    input_metadata = meta.all[,c('Group','Study',"Age","BMI","Country")]
    input_metadata$SampleID <- rownames(input_metadata)  
    meta.all <- input_metadata[input_metadata$SampleID %in% colnames(feat.all), ]
    feat.all <- feat.all[,colnames(feat.all) %in% input_metadata$SampleID ]
    if(groups=="CTR_CRC"){
      studies=c("AT-CRC","JPN-CRC","FR-CRC","CHN_WF-CRC","ITA-CRC","CHN_HK-CRC","CHN_SH-CRC","IND-CRC","US-CRC","DE-CRC")
    }else{
      studies=c("US-CRC-2","AT-CRC","JPN-CRC","FR-CRC","CN_WF-CRC","ITA-CRC","US-CRC-3","CHN-SH-CRC-2","CHN_SH-CRC-3")
    }
    for (study in studies){
      sample_table <- subset(meta.all, Study == study)
      row.names(sample_table) <- sample_table$SampleID
      feature_table <- feat.all[,colnames(feat.all) %in% row.names(sample_table)]
      sample_table <- sample_table[row.names(sample_table) %in% colnames(feature_table), ]
      sample_table$StudyID <- factor(sample_table $Study)
      filt_study_tab[["nonrare"]][[study]] <- feature_table
    }
  }

  filt_results <- list()
  for (groups in c("CTR_CRC")){
    if(groups=="CTR_CRC"){
      studies=c("AT-CRC","JPN-CRC","FR-CRC","CHN_WF-CRC","ITA-CRC","CHN_HK-CRC","CHN_SH-CRC","IND-CRC","US-CRC","DE-CRC")
    }else{
      studies=c("US-CRC-2","AT-CRC","JPN-CRC","FR-CRC","CN_WF-CRC","ITA-CRC","US-CRC-3","CHN-SH-CRC-2","CHN_SH-CRC-3")
    }
    for (study in studies){
      filt_results[[study]] <- read.csv(paste(feature_script_dir,analysis_level,"/",data_type, "/",groups,"/",study, "/adj_p_", groups, ".csv", sep=""), row.names = 1)
      filt_results[is.na(filt_results)] <- 1
    }
  }
  
  Calc_Other_community_metrics <- function(List_tab){
    Dataset_Char <- list()
    Dataset_Char[["nonrare"]] <- list()
    message("Calculating Sparsity")
    for(study in names(List_tab[["nonrare"]])){
      study_table <- List_tab[["nonrare"]][[study]]
      total_cells <- dim(study_table)[1] * dim(study_table)[2]
      Sparsity <- length(which(study_table==0))/total_cells
      Dataset_Char[["nonrare"]][[study]][["Sparsity"]] <- Sparsity
    }
    message("Calculating Number of Features")
    for(study in names(List_tab[["nonrare"]])){
      study_table <- List_tab[["nonrare"]][[study]]
      non_zero_rows <- study_table[rowSums(study_table != 0) > 0, ]
      num_feats <- dim(non_zero_rows)[1]
      Dataset_Char[["nonrare"]][[study]][["Number of Features"]] <- num_feats
    }
    message("Calculating median read depth")
    return(Dataset_Char)
  }
  filt_community_metrics <- Calc_Other_community_metrics(filt_study_tab)
  filt_nonrare_metrics_df <- do.call(rbind, filt_community_metrics[[1]])
  filt_nonrare_metrics_df1 <- data.frame(lapply(filt_nonrare_metrics_df, function(x) {
    as.numeric(as.vector(x))
  }))
  sig_counts <- data.frame(matrix(NA,
                                  nrow=length(names(filt_results)),
                                  ncol=ncol(filt_results[[1]]) + 1))
  rownames(sig_counts) <- names(filt_results)
  
  colnames(sig_counts) <- c("dataset", colnames(filt_results[[1]]))
  
  sig_counts$dataset <- rownames(sig_counts)
  
  filt_sig_counts <- sig_counts
  filt_sig_percent <- sig_counts
  
  
  for(study in rownames(filt_sig_counts)) {
    for(tool_name in colnames(filt_sig_counts)) {
      if(tool_name == "dataset") { next }
      
      if(! tool_name %in% colnames(filt_results[[study]])) {
        filt_sig_counts[study, tool_name] <- NA
        filt_sig_percent[study, tool_name] <- NA
        stop("Error tool name was not found in the result lookup table!!!!")
        next
      }
      if (tool_name=="lefse"){
        filt_sig_counts[study, tool_name] <- length(which(filt_results[[study]][, tool_name] < p_sig_lefse))
        filt_sig_percent[study, tool_name] <- (length(which(filt_results[[study]][, tool_name] < p_sig_lefse)) / dim(filt_study_tab[["nonrare"]][[study]])[1]) * 100
      }else{
        filt_sig_counts[study, tool_name] <- length(which(filt_results[[study]][, tool_name] < p_sig))
        filt_sig_percent[study, tool_name] <- (length(which(filt_results[[study]][, tool_name] < p_sig)) / dim(filt_study_tab[["nonrare"]][[study]])[1]) * 100
      }
      
    }
  }

  filt_mean_percent <- print(sort(colSums(filt_sig_percent[, -1], na.rm = TRUE) / (colSums(!is.na(filt_sig_percent[, -1])))))
  std_devs <- apply(filt_sig_percent[, -1], 2, function(x) sd(x, na.rm = TRUE))

  trans_=t(filt_sig_percent[, -1])
  filt_sig_percent_scaled <- data.frame(scale(trans_, center = TRUE, scale = TRUE))
  
  groups=="CTR_CRC"
  if(groups=="CTR_CRC"){
    studies=c("AT-CRC","JPN-CRC","FR-CRC","CHN_WF-CRC","ITA-CRC","CHN_HK-CRC","CHN_SH-CRC","IND-CRC","US-CRC","DE-CRC")
  }else{
    studies=c("US-CRC-2","AT-CRC","JPN-CRC","FR-CRC","CN_WF-CRC","ITA-CRC","US-CRC-3","CHN-SH-CRC-2","CHN_SH-CRC-3")
  }
  hackathon_metadata_filt <- data.frame(Study = studies)
  fixed_hackathon_metadata_filt <- hackathon_metadata_filt
  rownames(fixed_hackathon_metadata_filt) <- hackathon_metadata_filt$Study
  dataset_sample_sizes <- sapply(filt_study_tab$nonrare, ncol)
  fixed_hackathon_metadata_filt$log_N <- log(dataset_sample_sizes[fixed_hackathon_metadata_filt$Study])
  

  sort_filt_nonrare_metrics_df <- filt_nonrare_metrics_df[fixed_hackathon_metadata_filt$Study,,drop=F]
  fixed_hackathon_metadata_filt$Sparsity <- sort_filt_nonrare_metrics_df[,"Sparsity"]

  fixed_ds_names <- Data_set_names[colnames(filt_sig_percent_scaled)]
  colnames(filt_sig_percent_scaled) <- fixed_ds_names
  

  tool_rename <- tool_names[rownames(filt_sig_percent_scaled)]
  rownames(filt_sig_percent_scaled) <- tool_rename

  
  study_name <- gsub("-", ".", fixed_hackathon_metadata_filt$Study)

  rownames(fixed_hackathon_metadata_filt) <- Data_set_names[gsub("-", ".", rownames(fixed_hackathon_metadata_filt))]
  Alpha_genus_filt <- filt_sig_percent_scaled[,colnames(filt_sig_percent_scaled)]

  raw_count_df <- t(filt_sig_counts[,-1])
  rownames(raw_count_df) <- tool_names[rownames(raw_count_df)]

  
  colnames(raw_count_df) <- Data_set_names[gsub("-", ".", colnames(raw_count_df))]

  genus_raw_count_df <- raw_count_df[,colnames(Alpha_genus_filt)]

  identical(colnames(genus_raw_count_df), colnames(Alpha_genus_filt))
  

  Metadata_renames <- c(log_N="log(Sample size)", Sparsity="Sparsity")
  
  filt_metadata_col_to_rename <- which(colnames(fixed_hackathon_metadata_filt) %in% names(Metadata_renames))
  colnames(fixed_hackathon_metadata_filt)[filt_metadata_col_to_rename] <- Metadata_renames[colnames(fixed_hackathon_metadata_filt[filt_metadata_col_to_rename])]

  Alpha_genus_filt <- Alpha_genus_filt[sort(rownames(Alpha_genus_filt)), ]
  

  genus_raw_count_df <- genus_raw_count_df[sort(rownames(genus_raw_count_df)), ]
  
  fixed_hackathon_metadata_filt$Sparsity <- unlist(fixed_hackathon_metadata_filt$Sparsity)
  fixed_hackathon_metadata_filt$Richness <- unlist(fixed_hackathon_metadata_filt$Richness)
  my_colors <- colorRampPalette(c("#006DA3","#D1F0FF","white","#FFEBE0", "#FFDAC7","#FFA061"))(100)

  annotation_colors <- list(
    Sparsity = c("#CEAAD0", "#9355B0"),
    `log(Sample_size)` = c("#CFE2E6", "#FFA061") # Use backticks here
  )
  
  group="CTR_CRC"
  pdf(file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/single_study_heatmap.pdf",sep=""), width=5, height=4)

  pheatmap(t(Alpha_genus_filt),
           color = my_colors,
           legend = TRUE,
           display_numbers = t(genus_raw_count_df),
           treeheight_col = 0,
           cluster_cols = FALSE,
           cluster_rows = FALSE,
           main = analysis_level,
           show_colnames = TRUE,
           angle_col = "315")

  dev.off()
  write.table(x=filt_nonrare_metrics_df , file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_size.csv",sep=""), col.names = NA,
              row.names = T, quote=F, sep=",")
  write.table(x=filt_sig_percent, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_feat_persent.csv",sep=""), col.names = NA,
              row.names = T, quote=F, sep=",")
  write.table(x=Alpha_genus_filt, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_feat_scale_persent.csv",sep=""), col.names = NA,
              row.names = T, quote=F, sep=",")
  write.table(x=genus_raw_count_df, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_feat_count.csv",sep=""), col.names = NA,
              row.names = T, quote=F, sep=",")
  
  write.table(x=fixed_hackathon_metadata_filt, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_dataset_characteristics.csv",sep=""), col.names = NA,
              row.names=T, quote=F, sep=",")
  write.table(x=filt_mean_percent, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_filt_mean_percent.csv",sep=""), col.names = NA,
              row.names=T, quote=F, sep=",")
  write.table(x=std_devs, file=paste(figures_script_dir,analysis_level,"/",data_type,"/",group,"/",p_sig,"/singlestudy_filt_std_percent.csv",sep=""), col.names = NA,
              row.names=T, quote=F, sep=",")
}
