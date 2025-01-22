################################################################################
# Data Cleaning and Filtering Script for CTR_CRC Analysis
################################################################################

# Load required libraries
package_list <- c("tidyverse", "matrixStats", "vegan")
# Check and install missing packages
site <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN"
installed_packages <- rownames(installed.packages())
for (p in package_list) {
  if (!(p %in% installed_packages)) {
    install.packages(p, repos = site)
  }
}
# Clean environment
rm(list = ls())

# Log script start
cat('Starting data cleaning script\n')

# Record start time
start.time <- proc.time()[1]

# Set up directory paths
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir), "/Result/data/")
feature_script_dir <- paste0(dirname(script_dir), "/Result/feature/")

################################################################################
# Function Definition: CTR_CRC
################################################################################

CTR_CRC <- function(feature_file, output_file, group_info, study_num, mean_ab_th) {

  # Set memory limit (Windows only)
  memory.limit(size = 4000)

  # Load metadata and feature data
  meta.all <- read.csv(paste0(data_script_dir, "/metaA_all.csv",sep=""), sep = ',')
  feat.ab.all <- read.csv(feature_file, sep = ',', row.names = 1)
  feat.ab.all[is.na(feat.ab.all)] <- 0

  # Remove columns with all zero values
  feat.ab.all <- feat.ab.all[, colSums(feat.ab.all) != 0]

  # Filter metadata for relevant samples
  meta.all$Sample_ID <- as.character(meta.all$Sample_ID)
  meta.all <- meta.all[meta.all$Sample_ID %in% colnames(feat.ab.all), ]

  # Define studies based on group_info
  if ("ADA" %in% group_info) {
    crc.studies <- c("AT-CRC", "JPN-CRC", "FR-CRC", "CHN_WF-CRC", "ITA-CRC",
                     "CHN_SH-CRC-2", "CHN_SH-CRC-3", "US-CRC-2", "US-CRC-3")
  } else {
    crc.studies <- c("FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC",
                     "IND-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC")
  }

  # Filter metadata for training studies and groups
  meta.crc <- meta.all %>%
    filter(Study %in% crc.studies) %>%
    filter(Group %in% group_info)

  # Ensure feature matrix matches filtered metadata
  feat.ab.all <- as.matrix(feat.ab.all)
  stopifnot(all(meta.crc$Sample_ID %in% colnames(feat.ab.all)))

  # Extract species relative abundances for training set
  feat.rel.crc <- feat.ab.all[, meta.crc$Sample_ID]

  # Print summary of metadata
  print(table(meta.crc$Group))
  print(table(meta.crc$Study))
  print(table(meta.crc$Country))
  print(table(meta.crc$Study, meta.crc$Group))

  # Normalize features to proportions
  feat.rel.crc <- prop.table(feat.rel.crc, 2)

  # Log retained features before filtering
  f.feature <- sum(rowSums(feat.rel.crc) != 0)
  cat('Retaining', f.feature, 'features before low-abundance filtering...\n')

  # Calculate mean abundance for filtering
  temp.mean.ab <- t(sapply(row.names(feat.rel.crc),
                          FUN = function(marker) {
                            sapply(unique(meta.crc$Study),
                                   FUN = function(study, marker) {
                                     mean(feat.rel.crc[marker, which(meta.crc$Study == study)])
                                   },
                                   marker = marker)
                          }))

  # Filter features based on mean abundance across studies
  f.idx <- rowSums(temp.mean.ab >= mean_ab_th) >= study_num &
           row.names(feat.rel.crc) != '-1'
  feat.rel.red <- feat.rel.crc[f.idx, ]

  # Log retained features after filtering
  f.feature <- nrow(feat.rel.red)
  cat('Retaining', f.feature, 'features before quality control...\n')

  # Additional filtering function
  filter.f <- function(dat, Num) {
    SD <- apply(dat, 1, sd)
    num_0 <- apply(dat, 1, function(x) length(which(x == 0)))
    ave_abun <- apply(dat, 1, mean)
    tmp <- cbind(dat, data.frame(SD), data.frame(num_0), data.frame(ave_abun))
    colnames(tmp)[(ncol(dat) + 1):(ncol(dat) + 3)] <- c("sd", "count0", "avebun")
    dat_filter <- tmp[(tmp$count0 <= as.numeric(Num * 0.9)) & (tmp$sd > 0), ]
    return(dat_filter[, 1:ncol(dat)])
  }

  # Apply quality control filtering
  feat.filter <- t(filter.f(feat.rel.red, ncol(feat.rel.red)))
  cat('Retaining', ncol(feat.filter), 'features after quality control...\n')

  # Save filtered feature data
  fn.tax.rel.ab <- paste0(output_file, 'feature_rare_', paste(group_info, collapse = '_'), '.csv')
  if (!dir.exists(dirname(fn.tax.rel.ab))) {
    dir.create(dirname(fn.tax.rel.ab), recursive = TRUE)
  }
  write.table(feat.filter, file = fn.tax.rel.ab, quote = FALSE, sep = '\t',
              row.names = TRUE, col.names = TRUE)

  # Save filtered metadata
  meta_file <- paste0(output_file, 'meta_', paste(group_info, collapse = '_'), '.csv')
  write.table(meta.crc, file = meta_file, quote = FALSE, sep = '\t',
              row.names = TRUE, col.names = TRUE)
}

################################################################################
# End of Script
################################################################################
