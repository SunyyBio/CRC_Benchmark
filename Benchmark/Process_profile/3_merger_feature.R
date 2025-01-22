

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature/",sep="")


##############################################################
# Different feature selection tools merge features of all cohort
##############################################################
read_table_and_check_line_count <- function(filepath, ...) {
  
  exp_count <- as.numeric(sub(pattern = " .*$", "", system(command = paste("wc -l", filepath, sep=" "), intern = TRUE)))
  df <- read.table(filepath, ...)
  if(length(grep("^V", colnames(df))) != ncol(df)) {
    exp_count <- exp_count - 1
  }
  
  if(exp_count != nrow(df)) {
    stop(paste("Expected ", as.character(exp_count), " lines, but found ", as.character(nrow(df))))
  } else {
    return(df) 
  }
}

read_genera_hackathon_results <- function(feature_dir,groups) {
  
  da_tool_filepath <- list()
  da_tool_filepath[["ancom"]] <- paste(feature_dir,"All_features_tools/ANCOM/","Ancom_",group,".tsv", sep = "")
  da_tool_filepath[["RFECV"]] <- paste(feature_dir, "All_features_tools/RFECV.csv", sep = "")
  da_tool_filepath[["maaslin2"]] <- paste(feature_dir,"All_features_tools/Maaslin2/",group,"/all_results.tsv", sep = "")
  da_tool_filepath[["metagenomeSeq"]] <- paste(feature_dir,"All_features_tools/metagenomeSeq/metagenomeSeq_",group,".tsv", sep = "")
  da_tool_filepath[["ttest"]] <- paste(feature_dir,"All_features_tools/t_test/t_test_",group,".tsv", sep = "")
  da_tool_filepath[["wilcoxon"]] <- paste(feature_dir,"All_features_tools/wilcoxon/Wilcoxon_",group,".tsv", sep = "")
  da_tool_filepath[["lefse"]] <- paste(feature_dir,"All_features_tools/Lefse/lefse_",group,".csv", sep = "")
  
  adjP_colname <- list()
  adjP_colname[["ancom"]] <- "detected_0.8"
  adjP_colname[["RFECV"]] <- "RFECV"
  adjP_colname[["maaslin2"]] <- "pval"
  adjP_colname[["metagenomeSeq"]] <- "pvalues"
  adjP_colname[["ttest"]] <- "x"
  adjP_colname[["wilcoxon"]] <- "x"
  adjP_colname[["lefse"]] <- "P.unadj"

  da_tool_results <- list()
  missing_tools <- c()
  for(da_tool in names(da_tool_filepath)) {
    if(! (file.exists(da_tool_filepath[[da_tool]]))) {
      missing_tools <- c(missing_tools, da_tool)
      message(paste("File ", da_tool_filepath[[da_tool]], " not found. Skipping.", sep=""))
      next
    }
    if(da_tool %in% c("ancom")) {
      da_tool_results[["ancom"]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep="\t", row.names=2, header=TRUE,fill=TRUE,quote="")
    } else if(da_tool%in% c("RFECV")) {
       RFECV_data<- read_table_and_check_line_count(da_tool_filepath[["RFECV"]], sep=",", header=TRUE, stringsAsFactors=FALSE)
       target_column <- "FR.CRC_AT.CRC_ITA.CRC_JPN.CRC_CHN_WF.CRC_CHN_SH.CRC_CHN_HK.CRC_DE.CRC_IND.CRC_US.CRC_IFECV"
       if (!target_column %in% colnames(RFECV_data)) {
         stop(paste("Column", target_column, "not found in RFECV_data"))
       }
       RFECV_data <- data.frame(RFECV_data[[target_column]], stringsAsFactors = FALSE)
       colnames(RFECV_data) <- target_column
       valid_rows <- !is.na(RFECV_data[[target_column]]) & RFECV_data[[target_column]] != ""
       RFECV_data_filtered <- RFECV_data[valid_rows, , drop = FALSE]
       RFECV_data_filtered[[adjP_colname[["RFECV"]]]] <- 0
       da_tool_results[[da_tool]] <- RFECV_data_filtered
       rownames(da_tool_results[[da_tool]]) <- RFECV_data_filtered[[target_column]]
    }else if(da_tool%in% c("lefse")) {
      da_tool_results[[da_tool]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep=",", row.names=1,header=TRUE, stringsAsFactors=FALSE)
      da_tool_results[[da_tool]] <- da_tool_results[[da_tool]][da_tool_results[[da_tool]][["LDA"]] >= 2, , drop = FALSE]
      row_names <- rownames(da_tool_results[[da_tool]])
      row_names <- sub(".*\\|", "", row_names)
      rownames(da_tool_results[[da_tool]]) <- row_names
    } else {
      da_tool_results[[da_tool]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep="\t", row.names=1, header=TRUE,fill=TRUE,quote="")
    }
  }

  for(da_tool in names(da_tool_filepath)) {
    rownames(da_tool_results[[da_tool]]) <- gsub("\\.", "_", rownames(da_tool_results[[da_tool]]))
    rownames(da_tool_results[[da_tool]]) <- gsub("/", "_", rownames(da_tool_results[[da_tool]]))
  }
  
  all_rows <- c()
  for(da_tool in names(adjP_colname)) {
    all_rows <- c(all_rows, rownames(da_tool_results[[da_tool]]))
  }
  all_rows <- all_rows[-which(duplicated(all_rows))]
  adjP_table <- data.frame(matrix(NA, ncol=length(names(da_tool_results)), nrow=length(all_rows)))
  colnames(adjP_table) <- names(da_tool_results)
  rownames(adjP_table) <- all_rows
  for(da_tool in colnames(adjP_table)) {
    if(da_tool %in% missing_tools) {
      next
    }
    if(da_tool == "lefse") {
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <-da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]
    } else if(da_tool =="ancom") {
      sig_ancom_hits <- which(da_tool_results[[da_tool]][, adjP_colname[[da_tool]]])
      ancom_results <- rep(1, length(da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]))
      ancom_results[sig_ancom_hits] <- 0
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <- ancom_results
    } else {
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <- da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]
    }
  }
  return(list(raw_tables=da_tool_results,adjP_table=adjP_table))
}
for (class_dir in  c("class","order","family","genus","species","t_sgb","ko_gene")){
  group<-"CTR_CRC"
  data_type="Raw_log"
  feature_tool_dir <-paste(feature_script_dir,class_dir,"/",data_type,"/",group,"/",sep="")
  a <- read_genera_hackathon_results(feature_tool_dir, groups =group )
  raw_tables<-a$raw_tables
  adjP_table<-a$adjP_table
  write.csv(adjP_table,file=paste(feature_tool_dir,"/adj_p_",group,".csv",sep=""))
}


#############################################################################
## Different feature selection tools merge features of a single cohort

#############################################################################

read_table_and_check_line_count <- function(filepath, ...) {
  # Function to read in table and to check whether the row count equals the expected line count of the file.
  
  exp_count <- as.numeric(sub(pattern = " .*$", "", system(command = paste("wc -l", filepath, sep=" "), intern = TRUE)))
  df <- read.table(filepath, ...)
  if(length(grep("^V", colnames(df))) != ncol(df)) {
    exp_count <- exp_count - 1
  }
  if(exp_count != nrow(df)) {
    stop(paste("Expected ", as.character(exp_count), " lines, but found ", as.character(nrow(df))))
  } else {
    return(df) 
  }
}
read_genera_hackathon_results <- function(study_name,feature_dir,groups) {
  da_tool_filepath <- list()
  da_tool_filepath[["ancom"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/ANCOM/","Ancom_",group,".tsv", sep = "")
  da_tool_filepath[["maaslin2"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/Maaslin2/",group,"/all_results.tsv", sep = "")
  da_tool_filepath[["metagenomeSeq"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/metagenomeSeq/metagenomeSeq_",group,".tsv", sep = "")
  da_tool_filepath[["ttest"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/t_test/t_test_",group,".tsv", sep = "")
  da_tool_filepath[["wilcoxon"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/wilcoxon/wilcoxon_",group,".tsv", sep = "")
  da_tool_filepath[["lefse"]] <- paste(feature_dir,"/All_features_tools/",study_name,"/Lefse/lefse_",group,".csv", sep = "")
  da_tool_filepath[["RFECV"]] <- paste(feature_dir,"/All_features_tools/RFECV.csv", sep = "")
  
  adjP_colname <- list()
  study_name <- gsub("-", ".", study_name)
  adjP_colname[["ancom"]] <- "detected_0.8"
  adjP_colname[["RFECV"]] <- paste(study_name,"_IFECV",sep="")
  adjP_colname[["maaslin2"]] <- "pval"
  adjP_colname[["metagenomeSeq"]] <- "pvalues"
  adjP_colname[["ttest"]] <- "x"
  adjP_colname[["wilcoxon"]] <- "x"
  adjP_colname[["lefse"]] <- "P.unadj"

  # Read in results files and run sanity check that results files have expected number of lines
  da_tool_results <- list()
  missing_tools <- c()
  for(da_tool in names(da_tool_filepath)) {
    if(! (file.exists(da_tool_filepath[[da_tool]]))) {
      missing_tools <- c(missing_tools, da_tool)
      message(paste("File ", da_tool_filepath[[da_tool]], " not found. Skipping.", sep=""))
      next
    }
    if(da_tool %in% c("ancom")) {
      da_tool_results[[da_tool]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep="\t", row.names=2, header=TRUE,fill=TRUE,quote="")
    } else if(da_tool%in% c("RFECV")) {
      RFECV_data<- read_table_and_check_line_count(da_tool_filepath[["RFECV"]], sep=",", header=TRUE, stringsAsFactors=FALSE)
      target_column <- adjP_colname[["RFECV"]]
      if (!target_column %in% colnames(RFECV_data)) {
        stop(paste("Column", target_column, "not found in RFECV_data"))
      }
      RFECV_data <- data.frame(RFECV_data[[target_column]], stringsAsFactors = FALSE)
      new_col_name=paste(target_column,"_name",sep="")
      colnames(RFECV_data) <- new_col_name
      valid_rows <- !is.na(RFECV_data[[new_col_name]]) & RFECV_data[[new_col_name]] != ""
      RFECV_data_filtered <- RFECV_data[valid_rows, , drop = FALSE]
      RFECV_data_filtered[[adjP_colname[["RFECV"]]]] <- 0
      da_tool_results[[da_tool]] <- RFECV_data_filtered
      rownames(da_tool_results[[da_tool]]) <- RFECV_data_filtered[[new_col_name]]
    }else if(da_tool%in% c("lefse")) {
      da_tool_results[[da_tool]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep=",", row.names=1,header=TRUE, stringsAsFactors=FALSE)
      da_tool_results[[da_tool]] <- da_tool_results[[da_tool]][da_tool_results[[da_tool]][["LDA"]] >= 2, , drop = FALSE]
      row_names <- rownames(da_tool_results[[da_tool]])
      row_names <- sub(".*\\|", "", row_names)
      rownames(da_tool_results[[da_tool]]) <- row_names
    } else {
      da_tool_results[[da_tool]] <- read_table_and_check_line_count(da_tool_filepath[[da_tool]], sep="\t", row.names=1, header=TRUE,fill=TRUE,quote="")
    }
  }
  for(da_tool in names(da_tool_filepath)) {
    rownames(da_tool_results[[da_tool]]) <- gsub("\\.", "_", rownames(da_tool_results[[da_tool]]))
    rownames(da_tool_results[[da_tool]]) <- gsub("/", "_", rownames(da_tool_results[[da_tool]]))
  }
  all_rows <- c()
  for(da_tool in names(adjP_colname)) {
    all_rows <- c(all_rows, rownames(da_tool_results[[da_tool]]))
  }
  all_rows <- all_rows[-which(duplicated(all_rows))]
  adjP_table <- data.frame(matrix(NA, ncol=length(names(da_tool_results)), nrow=length(all_rows)))
  colnames(adjP_table) <- names(da_tool_results)
  rownames(adjP_table) <- all_rows
  for(da_tool in colnames(adjP_table)) {
    if(da_tool %in% missing_tools) {next}
    if(da_tool == "lefse") {
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <- da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]
    } else if(da_tool =="ancom") {
      sig_ancom_hits <- which(da_tool_results[[da_tool]][, adjP_colname[[da_tool]]])
      ancom_results <- rep(1, length(da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]))
      ancom_results[sig_ancom_hits] <- 0
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <- ancom_results
    } else {
      adjP_table[rownames(da_tool_results[[da_tool]]), da_tool] <- da_tool_results[[da_tool]][, adjP_colname[[da_tool]]]
    }
  }
  return(list(raw_tables=da_tool_results, adjP_table=adjP_table))
}
for (class_dir in  c("uniref_family")){
  for (group in c("CTR_CRC")){
    data_type="Raw_log"
    for(study_name in c('AT-CRC', 'JPN-CRC', 'FR-CRC',  'ITA-CRC', "CHN_HK-CRC","CHN_WF-CRC","CHN_SH-CRC", "IND-CRC", "US-CRC","DE-CRC")){
      
      feature_tool_dir <-paste(feature_script_dir,class_dir,"/",data_type,"/",group,"/",sep="")
      a <- read_genera_hackathon_results(study_name,feature_tool_dir, groups =group )
      raw_tables<-a$raw_tables
      adjP_table<-a$adjP_table
      # Ensure the directory exists before writing the file
      output_dir <- paste(feature_tool_dir, "/", study_name, sep = "")
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE) # Create the directory if it doesn't exist
      }
      
      # Write the CSV file
      output_file <- paste(output_dir, "/adj_p_", group, ".csv", sep = "")
      write.csv(adjP_table, file = output_file)
      
    }
  }
}




