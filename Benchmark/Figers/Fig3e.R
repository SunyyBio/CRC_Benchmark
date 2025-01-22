rm(list=ls())
# Define the list of required packages
package_list <- c("dplyr", "stats", "multcomp", "broom")
site <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN"
installed_packages <- rownames(installed.packages())
for (p in package_list) {
  if (!(p %in% installed_packages)) {
    install.packages(p, repos = site)
  }
}
lapply(package_list, library, character.only = TRUE)
# Initialize constants
log.n0 <- 1e-6
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
data_script_dir <- paste0(dirname(script_dir), "/Result/data/")
feature_script_dir <- paste0(dirname(script_dir), "/Result/feature/")
figures_script_dir <- paste0(dirname(script_dir), "/Result/figures/Fig03/")
for (feature_type in c("species")) {
  data_type = "Raw_log"
  group = "CTR_ADA"
  # Define study groups based on the selected group
  if (group == "CTR_CRC") {
    studies = c("FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC", "IND-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC", "DE-CRC")
  } else {
    studies = c("AT-CRC", "JPN-CRC", "FR-CRC", "CHN_WF-CRC", "ITA-CRC", "CHN_SH-CRC-2", "CHN_SH-CRC-3", "US-CRC-2", "US-CRC-3")
  }
  # Process each study
  for (study in studies) {
    results_df <- data.frame()
    memory.limit(size = 13000)
    meta.all <- read.csv(file = paste(data_script_dir, "meta.csv", sep = ""), sep = ',', stringsAsFactors = FALSE, header = TRUE, check.names = FALSE)
    rownames(meta.all) <- meta.all$Sample_ID
    meta.all <- subset(meta.all, Study == study)
    meta.all$StudyID <- factor(meta.all$Study)
    feat.abu <- read.csv(paste0(data_script_dir, feature_type, "/feature_rare_", group, ".csv"), sep = '\t', stringsAsFactors = FALSE, header = TRUE, row.names = 1, check.names = FALSE)
    feat.abu <- as.data.frame(t(feat.abu))
    feature <- read.csv(paste0(feature_script_dir, feature_type, "/", data_type, "/", group, "/feature.csv"))
    if (group == "CTR_CRC") {
      feature <- feature$FR.CRC_AT.CRC_ITA.CRC_JPN.CRC_CHN_WF.CRC_CHN_SH.CRC_CHN_HK.CRC_DE.CRC_IND.CRC_US.CRC_rf_optimal
    } else {
      feature <- feature$AT.CRC_JPN.CRC_FR.CRC_CHN_WF.CRC_ITA.CRC_CHN_SH.CRC.2_US.CRC.3_CHN_SH.CRC.4_US.CRC.2_rf_optimal
    }
    feature <- feature[feature != ""]
    feature <- as.character(feature)
    # Filter feature abundance data based on selected features
    feat.abu <- feat.abu[feature, ]
    meta.all <- meta.all[rownames(meta.all) %in% colnames(feat.abu), ]
    feat.abu <- feat.abu[, rownames(meta.all)]
    feat.abu[is.na(feat.abu)] <- 0
    # Normalize feature abundance data
    colSums_vec <- colSums(feat.abu)
    feat.abu <- sweep(feat.abu, 2, colSums_vec, FUN = "/")
    # Create a data frame for alpha diversity analysis
    alpha_df <- data.frame(t(feat.abu), Group = meta.all[colnames(feat.abu), ]$Group)
    # Calculate fold change values for each species
    if (group == "CTR_CRC") {
      for (species in rownames(feat.abu)) {
        CRC <- subset(alpha_df, Group == "CRC")[, species, drop = FALSE]
        CTR <- subset(alpha_df, Group == "CTR")[, species, drop = FALSE]
        q.p <- quantile(log10(CRC + log.n0), probs = seq(0.1, 0.9, 0.1), na.rm = TRUE)
        q.n <- quantile(log10(CTR + log.n0), probs = seq(0.1, 0.9, 0.1), na.rm = TRUE)
        # Calculate generalized fold change
        fc_value <- (sum(q.p - q.n)) / length(q.p)
        results_df <- rbind(results_df, data.frame(ClassDir = feature_type, Species = species, FC_Value = fc_value))
      }
    } else {
      for (species in rownames(feat.abu)) {
        ADA <- subset(alpha_df, Group == "ADA")[, species, drop = FALSE]
        CTR <- subset(alpha_df, Group == "CTR")[, species, drop = FALSE]
        q.p <- quantile(log10(ADA + log.n0), probs = seq(0.1, 0.9, 0.1), na.rm = TRUE)
        q.n <- quantile(log10(CTR + log.n0), probs = seq(0.1, 0.9, 0.1), na.rm = TRUE)
        fc_value <- (sum(q.p - q.n)) / length(q.p)
        results_df <- rbind(results_df, data.frame(ClassDir = feature_type, Species = species, FC_Value = fc_value))
      }
    }
    # Save results to a CSV file
    output_path <- paste(figures_script_dir, feature_type,"/", data_type, "/",group,"/", study, "_FC_feature_study.csv", sep = "")
    write.csv(results_df, output_path, row.names = FALSE)
  }
}


######分析AUC与什么相关
df=read.csv("F:/bioinfoclub/benchmark/result/第一部分的结果/结果/模型层级数据比较/工作簿2.csv",fileEncoding = "UTF-8")

# 运行Kruskal-Wallis测试
kruskal_group <- kruskal.test(reformulate("feature", response = "AUC"), data = df)
p_value_group <- -log(kruskal_group$p.value)
chi_sq_group <- kruskal_group$statistic

