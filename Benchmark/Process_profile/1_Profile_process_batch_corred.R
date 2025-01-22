################################################################################
# Meta-Analysis Script Using MMUPHin
################################################################################
# Define the list of required packages
package_list <- c("MMUPHin", "magrittr", "dplyr", "ggplot2", "labdsv", "vegan")
# Check and install missing packages
site <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN"
installed_packages <- rownames(installed.packages())
for (p in package_list) {
  if (!(p %in% installed_packages)) {
    install.packages(p, repos = site)
  }
}
# Load all required packages
lapply(package_list, library, character.only = TRUE)
# Clean environment
rm(list = ls())

################################################################################
# Function Definition: MMUPHin_info
################################################################################

MMUPHin_info <- function(class_dir, groups) {

  # Define directory paths
  script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  data_script_dir <- paste0(dirname(script_dir), "/Result/data/")
  feature_script_dir <- paste0(dirname(script_dir), "/Result/feature/")

  # Set memory limit (Windows only)
  memory.limit(size = 130000)

  # Load metadata
  meta.all <- read.csv(
    file = paste0(data_script_dir, "meta.csv"),
    sep = ',',
    stringsAsFactors = FALSE,
    header = TRUE,
    check.names = FALSE
  )

  # Define studies based on group
  if (groups == "CTR_CRC") {
    studies <- c("FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC",
                 "IND-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC")
  } else {
    studies <- c("AT-CRC", "JPN-CRC", "FR-CRC", "CHN_WF-CRC", "ITA-CRC",
                 "CHN_SH-CRC-2", "CHN_SH-CRC-3", "US-CRC-2", "US-CRC-3")
  }

  # Filter metadata for selected studies
  meta.all <- subset(meta.all, Study %in% studies)
  rownames(meta.all) <- meta.all$Sample_ID

  # Load feature data
  feat.all <- read.csv(
    file = paste0(data_script_dir, class_dir, "/feature_rare_", groups, ".csv"),
    sep = '\t',
    stringsAsFactors = FALSE,
    header = TRUE,
    row.names = 1,
    check.names = FALSE
  )

  # Ensure common samples between metadata and feature data
  common_samples <- intersect(row.names(meta.all), row.names(feat.all))
  feat.all <- feat.all[common_samples, ]
  meta.all <- meta.all[common_samples, ]

  # Transpose feature data
  feat.abu <- t(feat.all)
  meta.all$StudyID <- factor(meta.all$Study)

  # Replace NA values with 0
  feat.abu[is.na(feat.abu)] <- 0

  # Normalize feature data to proportions
  feat.abu <- apply(feat.abu, 2, function(x) x / sum(x, na.rm = TRUE))

  # Define output directory and diagnostic plot filename
  output_dir <- paste0(data_script_dir, class_dir, '/', groups, '/', sep = "")
  filename <- paste0(output_dir, groups, "_adjust_batch_diagnostic.pdf")

  # Create directories if they do not exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Adjust for batch effects
  fit_adjust_batch <- adjust_batch(
    feature_abd = feat.abu,
    batch = "StudyID",
    data = meta.all,
    covariates = "Group",
    control = list(rma_method = "HS", transform = "AST", diagnostic_plot = filename)
  )

  # Save batch-adjusted feature data
  CRC_abd_adj <- fit_adjust_batch$feature_abd_adj
  write.csv(
    CRC_abd_adj,
    file = paste0(data_script_dir, class_dir, '/', groups, "_adj_batch.csv")
  )
}

################################################################################
# End of Script
################################################################################
