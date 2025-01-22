# Define the list of required packages
package_list <- c("magrittr", "dplyr", "ggplot2", "vegan", "labdsv", "limma", "tibble", "cowplot")
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
rm(list=ls())

script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir),"/Result/data",sep="")
feature_script_dir <- paste0(dirname(script_dir),"/Result/feature",sep="")
figures_script_dir <- paste0(dirname(script_dir),"/Result/figures/Adnois/",sep="")

# Function to create directory if it doesn't exist
create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("Directory created:", dir_path, "\n")
  } else {
    cat("Directory already exists:", dir_path, "\n")
  }
}
# Create directories if they do not exist
create_dir_if_not_exists(data_script_dir)
create_dir_if_not_exists(figures_script_dir)


### Meta-analysis and Adonis analysis
for (class_dir in c("genus","species","t_sgb")) {
  for (groups in c("CTR_CRC")) {
    for (data_type in c("Raw","Raw_log","Batch")) {
      # Load metadata
      meta.all <- read.csv(
        file = file.path(data_script_dir, "meta.csv"),
        sep = ',', stringsAsFactors = FALSE, header = TRUE, row.names = 2, check.names = FALSE
      )
      
      # Filter metadata for specific studies
      meta.all <- subset(meta.all, Study %in% c(
        'AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC',
        'DE-CRC', 'CHN_HK-CRC', 'CHN_SH-CRC', 'IND-CRC', 'US-CRC'
      ))
      
      # Preprocess metadata
      meta.all$StudyID <- factor(meta.all$Study)
      meta.all$Gender <- gsub("female", "F", meta.all$Gender)
      meta.all$Gender <- gsub("male", "M", meta.all$Gender)
      
      # Remove empty columns
      meta.all <- meta.all[, colnames(meta.all) != ""]
      
      # Subset metadata for different categories
      meta_all_gender <- meta.all[!is.na(meta.all$Gender), ]
      meta_all_Age <- meta.all[!is.na(meta.all$Age), ]
      meta_all_BMI <- meta.all[!is.na(meta.all$BMI), ]
      
      # Group BMI into categories
      meta_all_BMI$BMI_Group <- cut(
        meta_all_BMI$BMI,
        breaks = c(-Inf, 25, 30, Inf),
        labels = c("Normal", "Overweight", "Obese"),
        right = TRUE
      )
      meta_all_BMI$BMI_Group <- factor(meta_all_BMI$BMI_Group)
      
      # Load feature data based on data type
      if (data_type == "Raw") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, paste0("feature_rare_", groups, ".csv")),
          sep = '\t', stringsAsFactors = FALSE, header = TRUE
        )
        feat.all <- as.data.frame(t(feat.all))
      } else if (data_type == "Raw_log") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, paste0("feature_rare_", groups, ".csv")),
          sep = '\t', stringsAsFactors = FALSE, header = TRUE
        )
        feat.all <- apply(feat.all + 1, 2, log)
        feat.all <- as.data.frame(t(feat.all))
      } else if (data_type == "Batch") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, "CTR_CRC_adj_batch.csv"),
          sep = ',', stringsAsFactors = FALSE, header = TRUE, row.names = 1, check.names = FALSE
        )
      }
      
      # Filter common rows
      common_rows_all <- intersect(colnames(feat.all), rownames(meta.all))
      feat.all_filter <- feat.all[, common_rows_all]
      meta.all_filter <- meta.all[common_rows_all, ]
      
      common_gender <- intersect(colnames(feat.all), rownames(meta_all_gender))
      feat_all_gender <- feat.all[,common_gender]
      meta_all_gender <- meta_all_gender[common_gender, ]
      
      common_Age <- intersect(colnames(feat.all), rownames(meta_all_Age))
      feat_all_Age <- feat.all[,common_Age]
      meta_all_Age <- meta_all_Age[common_Age, ]
      
      
      common_BMI <- intersect(colnames(feat.all), rownames(meta_all_BMI))
      feat_all_BMI <- feat.all[,common_BMI]
      meta_all_BMI <- meta_all_BMI[common_BMI, ]
    
      # Calculate Bray-Curtis distance matrix
      D_All <- vegdist(t(feat.all_filter), method = "bray")
      D_gender <- vegdist(t(feat_all_gender),method = "bray")
      D_Age <- vegdist(t(feat_all_Age),method = "bray")
      D_BMI <- vegdist(t(feat_all_BMI),method = "bray")

    
      # Perform Adonis analysis
      fit_adonis_study <- adonis2(D_All ~ Study, permutations = 999, data = meta.all_filter)
      fit_adonis_group <- adonis2(D_All ~ Group, permutations = 999, data = meta.all_filter)
      fit_adonis_gender <- adonis2(D_gender ~ Gender, permutations = 999, data = meta_all_gender)
      fit_adonis_Age <- adonis2(D_Age ~ Age, permutations = 999, data = meta_all_Age)
      fit_adonis_BMI <- adonis2(D_BMI ~ BMI, permutations = 999, data = meta_all_BMI)

      # Save results to CSV
      df_adonis_result <- as.data.frame(matrix(nrow = 5, ncol = 5, 0))
      rownames(df_adonis_result) <- c("Study", "Group","Gender","Age","BMI")
      colnames(df_adonis_result) <- c("Df", "SumOfSqs", "R2", "F", "Pr(>F)")
      df_adonis_result["Study", ] <- as.list(fit_adonis_study["Study", ])
      df_adonis_result["Group", ] <- as.list(fit_adonis_group["Group", ])
      df_adonis_result["Gender", ] <- as.list(fit_adonis_gender["Gender", ])
      df_adonis_result["Age", ] <- as.list(fit_adonis_Age["Age", ])
      df_adonis_result["BMI", ] <- as.list(fit_adonis_BMI["BMI", ])
      write.csv(
        df_adonis_result,
        file = file.path(figures_script_dir, paste0(class_dir, "_", groups, "_", data_type, "_adonis.csv"))
      )
    }
  }
}

# Perform Adonis analysis based on selected features
for (class_dir in c("genus","species","t_sgb")) {
  for (groups in c("CTR_CRC")) {
    for (data_type in c("Raw", "Raw_log", "Batch")) {
      # Set memory limit for processing
      memory.limit(size = 13000)
      significance_threshold <- 0.001
      
      # Load metadata
      meta.all <- read.csv(
        file = file.path(data_script_dir, "metaA_all.csv"),
        sep = ',', stringsAsFactors = FALSE, header = TRUE, check.names = FALSE
      )
      rownames(meta.all) <- meta.all$Sample_ID
      meta.all$StudyID <- factor(meta.all$Study)
      
      # Load feature data based on data type
      if (data_type == "Raw") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, paste0("feature_rare_", groups, ".csv")),
          sep = '\t', stringsAsFactors = FALSE, header = TRUE
        )
        feat.all <- as.data.frame(t(feat.all))
      } else if (data_type == "Raw_log") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, paste0("feature_rare_", groups, ".csv")),
          sep = '\t', stringsAsFactors = FALSE, header = TRUE
        )
        feat.all <- apply(feat.all + 1, 2, log)
        feat.all <- as.data.frame(t(feat.all))
      } else if (data_type == "Batch") {
        feat.all <- read.csv(
          file = file.path(data_script_dir, class_dir, "CTR_CRC_adj_batch.csv"),
          sep = ',', stringsAsFactors = FALSE, header = TRUE, row.names = 1, check.names = FALSE
        )
      }
      
      # Filter common rows between features and metadata
      common_rows_all <- intersect(colnames(feat.all), rownames(meta.all))
      feat.all <- feat.all[, common_rows_all]
      meta.all <- meta.all[common_rows_all, ]
      feat.all[is.na(feat.all)] <- 0
     
      # Load and filter significant features
      feature <- read.csv(
        file = file.path(feature_script_dir, class_dir, data_type, groups, "All_features_tools/Maaslin2/",groups,"/all_results.tsv"),
        sep = '\t')
      filtered_features <- feature[feature[["qval"]] <= significance_threshold, ]
    
      selected_row_names <-filtered_features$feature
      common_rows1 <- intersect(selected_row_names, rownames(feat.all))
      
      feat.abu1 <- feat.all[common_rows1, ]
      feat.abu1 <- feat.abu1[, colSums(feat.abu1)!= 0]
      meta.all <- meta.all[colnames(feat.abu1),]


      # Calculate Bray-Curtis distance matrix
      D_feature <- vegdist(t(feat.abu1), method = "bray", na.rm = TRUE)
      
      # Perform Adonis analysis
      fit_adonis_feature_study_ID <- adonis2(D_feature ~ Study, permutations = 999, data = meta.all)
      fit_adonis_feature_Group <- adonis2(D_feature ~ Group, permutations = 999, data = meta.all)
      
      # Save results to CSV
      df_feature_result <- data.frame(
        Df = c(fit_adonis_feature_Group[["Group", "Df"]], fit_adonis_feature_study_ID[["Study", "Df"]]),
        SumOfSqs = c(fit_adonis_feature_Group[["Group", "SumOfSqs"]], fit_adonis_feature_study_ID[["Study", "SumOfSqs"]]),
        R2 = c(fit_adonis_feature_Group[["Group", "R2"]], fit_adonis_feature_study_ID[["Study", "R2"]]),
        F = c(fit_adonis_feature_Group[["Group", "F"]], fit_adonis_feature_study_ID[["Study", "F"]]),
        `Pr(>F)` = c(fit_adonis_feature_Group[["Group", "Pr(>F)"]], fit_adonis_feature_study_ID[["Study", "Pr(>F)"]])
      )
      rownames(df_feature_result) <- c("Group", "Study")
      
      write.csv(
        df_feature_result,
        file = file.path(figures_script_dir, paste0(class_dir, "_", groups, "_", data_type, "_MaAsLin_adnois.csv"))
      )
    }
  }
}


# Define the script and data directories
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- file.path(dirname(script_dir), "Result/data")
figures_script_dir <- file.path(dirname(script_dir), "Result/figures/Pcoa")

# Function to create directories if they do not exist
create_dir_if_not_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("Directory created:", dir_path, "\n")
  } else {
    cat("Directory already exists:", dir_path, "\n")
  }
}

# Ensure the output directory exists
create_dir_if_not_exists(figures_script_dir)

# Define group labels and color mappings
new_labels <- c("CTR", "CRC")
new_colors <- c("#CCCC00", "#FF7A7A")
new_color_mapping <- setNames(new_colors, new_labels)

# Define study labels and colors
labels <- c("AT-CRC", "FR-CRC", "JPN-CRC", "CHN_WF-CRC", "ITA-CRC", "US-CRC", "DE-CRC", "IND-CRC", "CHN_SH-CRC", "CHN_HK-CRC")
colors <- c("#1F77B4", "#FF5500", "#2CA02C", "#CC0044", "#6A0DAD", "#A0522D", "#E6E600", "#000000", "#DD33FF", "#004BBD", "#00CED1")
color_mapping <- setNames(colors, labels)

# Perform PCoA analysis for each class directory
for (class_dir in c("class", "order", "family", "genus", "species", "t_sgb", "ko_gene", "uniref_family")) {
  groups <- "CTR_CRC"
  
  # Load feature data
  feature_file <- file.path(data_script_dir, class_dir, paste0("feature_rare_", groups, ".csv"))
  feat.all <- read.csv(feature_file, sep = '\t', stringsAsFactors = FALSE, header = TRUE)
  feat.all <- as.data.frame(t(feat.all))
  feat.all <- replace(feat.all, is.na(feat.all), 0)
  
  # Load metadata
  meta_file <- file.path(data_script_dir, "meta.csv")
  meta <- read.csv(meta_file, sep = ',', stringsAsFactors = FALSE, header = TRUE, check.names = FALSE)
  rownames(meta) <- meta$Sample_ID
  
  # Filter metadata for selected groups and studies
  group_class <- c("CTR", "CRC")
  meta <- meta[meta$Group %in% group_class & meta$Study %in% labels, ]
  
  # Match features with metadata
  data <- feat.all[, colnames(feat.all) %in% rownames(meta)]
  meta <- meta[rownames(meta) %in% colnames(data), ]
  
  # Skip iteration if no valid data
  if (nrow(meta) == 0 || ncol(data) == 0) {
    message("No valid data for analysis, skipping...")
    next
  }
  
  # Calculate Bray-Curtis distance matrix
  dist_all <- vegdist(t(data), na.rm = TRUE, method = "bray")
  
  # Perform PCoA
  pco_results <- pco(dist_all, k = 2)
  
  # Prepare axis titles
  axis_1_title <- paste0("PCo1 [", round((pco_results$eig[1] / sum(pco_results$eig)) * 100, 1), "%]")
  axis_2_title <- paste0("PCo2 [", round((pco_results$eig[2] / sum(pco_results$eig)) * 100, 1), "%]")
  
  # Create a data frame for plotting
  df_plot <- tibble(
    Axis1 = -1 * pco_results$points[, 1],
    Axis2 = pco_results$points[, 2],
    Sample_ID = rownames(pco_results$points),
    Group = meta$Group,
    Study = meta$Study,
    Country = meta$Country
  )
  
  # Generate PCoA plot
  g_main <- df_plot %>%
    ggplot(aes(x = Axis1, y = Axis2, shape = Group, col = Study)) +
    geom_point(size = 2) +
    scale_colour_manual(values = color_mapping, name = "Study") +
    scale_shape_manual(values = c("CTR" = 1, "CRC" = 2), name = "Group") +
    xlab(axis_1_title) + 
    ylab(axis_2_title) +
    theme(
      panel.background = element_rect(fill = 'white', color = 'black'),
      axis.ticks = element_line(color = 'black'),
      axis.text = element_text(size = 10, color = 'black'),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      panel.grid = element_blank(),
      legend.position = "right"
    )
  
  # Save the plot to a PDF file
  pdf_path <- file.path(figures_script_dir, paste0(class_dir,"_CTR_CRC_Raw_pcoa.pdf"))
  
  tryCatch({
    ggsave(filename = pdf_path, plot = g_main, device = "pdf", width = 8, height = 6, units = "in")
    message("PDF file saved:", pdf_path)
  }, error = function(e) {
    message("Error saving PDF file:", e$message)
  })
}

