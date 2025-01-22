################################################################################
# Microbiome Data Preprocessing and Analysis Script
################################################################################
# Clean environment
rm(list = ls())
# Required libraries
required_packages <- c("dplyr", "ggplot2", "Maaslin2", "microeco", "magrittr", "pacman", "tidyverse")

# Function to install and load missing packages
install_missing_packages <- function(package_list, repo = "https://mirrors.tuna.tsinghua.edu.cn/CRAN") {
  for (pkg in package_list) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg, repos = repo)
      library(pkg, character.only = TRUE, quietly = TRUE)
    }
  }
}

# Define directories
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir), "/Result/data/")
feature_script_dir <- paste0(dirname(script_dir), "/Result/feature/")

################################################################################
# Main Function: analyze_microbiome_data
################################################################################

analyze_microbiome_data <- function(meta_file, feat_file, group, class_dir, output_dir, studies_of_interest, taxonomic_level = "species", data_type = "Raw_log", apply_log = TRUE) {
  
  # Install missing packages
  install_missing_packages(required_packages)
  
  # Load metadata
  meta.all <- read.csv(
    file = meta_file,
    sep = ',',
    stringsAsFactors = FALSE,
    header = TRUE
  )
  
  # Function to detect the separator for feature files
  detect_separator <- function(file_path) {
    first_line <- readLines(file_path, n = 1)
    if (grepl(",", first_line)) {
      return(",")
    } else if (grepl("\t", first_line)) {
      return("\t")
    } else {
      stop("Unable to detect the separator. Please check the file format.")
    }
  }
  
  # Load feature data
  sep <- detect_separator(feat_file)
  feat.all <- read.csv(
    file = feat_file,
    sep = sep,
    stringsAsFactors = FALSE,
    header = TRUE
  )
  
  # Transpose feature data if needed
  if (!any(colnames(feat.all) %in% meta.all$Sample_ID)) {
    feat.all <- as.data.frame(t(feat.all))
  }
  
  # Set rownames if the first column contains IDs
  if (is.character(feat.all[[1]])) {
    rownames(feat.all) <- feat.all[[1]]
    feat.all <- feat.all[, -1, drop = FALSE]
  }
  
  # Filter metadata
  meta.all <- meta.all[, c("Sample_ID", "Study", "Group")]
  meta.all.filtered <- meta.all[meta.all$Study %in% studies_of_interest, ]
  row.names(meta.all.filtered) <- meta.all.filtered$Sample_ID
  
  # Apply log transformation if required
  if (apply_log) {
    feat.all <- apply(feat.all + 1, 2, log)
  }
  feat.all <- as.data.frame(feat.all)
  
  # Find common samples between metadata and feature data
  common_samples <- intersect(row.names(meta.all.filtered), colnames(feat.all))
  meta.all.filtered <- meta.all.filtered[common_samples, ]
  feat.all <- feat.all[, common_samples]
  
  # Normalize feature data to proportions
  feat.all <- apply(feat.all, 2, function(x) x / sum(x, na.rm = TRUE))
  
  # Function to create output folders
  create_output_folder <- function(path) {
    if (!dir.exists(path)) {
      success <- dir.create(path, recursive = TRUE)
      if (!success) {
        stop(sprintf("Unable to create directory: %s", path))
      }
    }
  }
  
  # Create output directories
  create_output_folder(output_dir)
  create_output_folder(file.path(output_dir, "Lefse"))
  create_output_folder(file.path(output_dir, "Maaslin2", group))
  create_output_folder(file.path(output_dir, "Maaslin2", group, "figures"))
  
  # Perform Maaslin2 analysis
  fit_data <- Maaslin2(
    input_data = feat.all,
    input_metadata = meta.all.filtered,
    transform = "none",
    output = file.path(output_dir, "Maaslin2", group),
    fixed_effects = c("Group"),
    correction = "BH"
  )
  
  # Lefse analysis
  tax <- data.frame(
    kingdom = NA, 
    Phylum = NA, 
    Class = NA, 
    Order = NA, 
    Family = NA, 
    Genus = NA, 
    Species = row.names(feat.all)
  )
  row.names(tax) <- row.names(feat.all)
  dataset <- microtable$new(
    sample_table = meta.all.filtered,
    otu_table = as.data.frame(feat.all),
    tax_table = tax
  )
  lefse <- trans_diff$new(
    dataset = dataset,
    method = "lefse",
    group = "Group",
    alpha = 1,
    taxa_level = "species",
    p_adjust_method = "fdr"
  )
  
  # Save Lefse results
  write.table(
    lefse$res_diff,
    file.path(output_dir, "Lefse", paste0("lefse_", group, ".csv")),
    sep = ','
  )
  
  # Function to run external analysis tools
  run_analysis_tools <- function(feat.all, meta.all.filtered, output_dir) {
    analysis_tools <- list(
      "Wilcoxon" = paste(script_dir, "/Tool_script/Run_Wilcox.R", sep = ""),
      "metagenomeSeq" = paste(script_dir, "/Tool_script/Run_metagenomeSeq.R", sep = ""),
      "t_test" = paste(script_dir, "/Tool_script/Run_t_test.R", sep = ""),
      "ANCOM" = paste(script_dir, "/Tool_script/Run_ANCOM.R", sep = "")
    )
    
    for (method in names(analysis_tools)) {
      tool_script <- analysis_tools[[method]]
      output_path <- normalizePath(file.path(output_dir, method, paste(method, "_", group, ".tsv", sep = "")), mustWork = FALSE)
      dir_path <- dirname(output_path)
      
      # Create directories if needed
      if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
        cat("Directory created: ", dir_path, "\n")
      }
      
      # Save temporary files
      feat_file <- paste0(feature_script_dir, class_dir, "/", data_type, "/feat.csv")
      meta_file <- paste0(feature_script_dir, class_dir, "/", data_type, "/meta.csv")
      write.csv(feat.all, feat_file, row.names = TRUE)
      write.csv(meta.all.filtered, meta_file, row.names = TRUE)
      
      tryCatch({
        if (method == "ANCOM") {
          ANCOM_DIR <- paste(script_dir, "/Tool_script/Ancom2_Script", sep = "")
          ANCOM_Script_Path <- file.path(ANCOM_DIR, "ancom_v2.1.R")
          cmd <- sprintf("Rscript \"%s\" \"%s\" \"%s\" \"%s\" \"%s\"", tool_script, feat_file, meta_file, output_path, ANCOM_Script_Path)
        } else {
          cmd <- sprintf("Rscript \"%s\" \"%s\" \"%s\" \"%s\"", tool_script, feat_file, meta_file, output_path)
        }
        
        # Execute command
        system(cmd)
        cat(sprintf("Method %s executed successfully.\n", method))
      }, error = function(e) {
        cat(sprintf("Error executing method %s: %s\n", method, e$message))
        stop(sprintf("Execution halted: Method %s failed.", method))
      })
    }
  }
  
  # Run external analysis tools
  run_analysis_tools(feat.all, meta.all.filtered, output_dir)
}

################################################################################
# Run Analysis for All Studies
################################################################################

for (class_dir in c("class")) {
  group <- "CTR_CRC"
  data_type <- "Raw"
  studies_of_interest <- c(
    'AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "DE-CRC",
    "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC", "US-CRC"
  )
  meta_file <- paste0(data_script_dir, "/meta.csv")
  feat_file <- paste0(data_script_dir, class_dir, "/feature_rare_", group, ".csv")
  output_dir <- paste0(feature_script_dir, class_dir, "/", data_type, "/", group, "/All_features_tools")
  analyze_microbiome_data(
    meta_file, feat_file, group, class_dir, output_dir, studies_of_interest,
    data_type, apply_log = FALSE
  )
}


################################################################################
# Run Analysis for single Study
################################################################################
for (class_dir in  c("class")){
  group <- "CTR_CRC"
  data_type="Raw_log"
  studies_of_interests <- c('AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "DE-CRC", "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC", "US-CRC")
  for (studies_of_interest in studies_of_interests){
    meta_file <- paste0(data_script_dir, "/meta.csv",sep="")
    feat_file <- paste(data_script_dir, class_dir,"/feature_rare_",group,".csv", sep="")
    output_dir=paste(feature_script_dir,class_dir,"/",data_type,"/",group,"/All_features_tools/",studies_of_interest,sep='')
    analyze_microbiome_data(meta_file,feat_file, group, class_dir, output_dir, studies_of_interest,data_type,apply_log =FALSE)
  }
}

#运行batch的数据
for (class_dir in  c("class","order","family","genus","species","t_sgb","ko_gene","uniref_family")){
  group <- "CTR_CRC"
  data_type="Batch"
  studies_of_interest <- c('AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "DE-CRC", "CHN_HK-CRC", "CHN_SH-CRC", "IND-CRC", "US-CRC")
  meta_file <- paste0(data_script_dir, "/meta.csv",sep="")
  feat_file <- paste(data_script_dir, class_dir,"/CTR_CRC_adj_batch.csv", sep="")
  output_dir=paste(feature_script_dir,class_dir,"/",data_type,"/",group,"/All_feature_tools/",studies_of_interest,sep='')
  analyze_microbiome_data(meta_file,feat_file, group, class_dir, output_dir, studies_of_interest,data_type,apply_log =FALSE)
}

#运行batch的数据
for (class_dir in  c("ko_gene")){
  group <- "CTR_CRC"
  studies_of_interests <- c("DE-CRC")
  for (studies_of_interest in studies_of_interests){
  meta_file <- "C:/Users/sunny/Desktop/论文/meta3.csv"
  feat_file <- paste("G:/deeplearning/CRC/benchmarker/",class_dir,"/Result/MMUPHin/new3/CTR_CRC/CTR_CRC_adj_batch.csv", sep="")
  output_dir=paste("G:/deeplearning/CRC/benchmarker/",class_dir,"/Result/feature_selection/new3/Batch/All_features_tools/",studies_of_interest,sep='')
  analyze_microbiome_data(meta_file,feat_file, group, class_dir, output_dir, studies_of_interest, apply_log = FALSE)
  }
}




