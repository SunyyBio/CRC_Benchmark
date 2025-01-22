import argparse
import csv
import os
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from Meta_iTL.function.MNN_function import split_data
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path
data_path = sys.path[1] + '\\Result\\data\\'
figure_path = sys.path[1] + '\\Result\\figures\\'

def find_top_half_nearest_cosine_neighbors(df1, df2,dis_cosin):

    distances = cdist(df1.iloc[:, 2:].values, df2.iloc[:, 2:].values, metric='cosine')
    num_neighbors_df1 = int(dis_cosin* df2.shape[0])
    nearest_neighbors_df1 = np.argsort(distances, axis=1)[:, :num_neighbors_df1]
    num_neighbors_df2 = int(dis_cosin* df1.shape[0])
    nearest_neighbors_df2 = np.argsort(distances.T, axis=1)[:, :num_neighbors_df2]
    distance_threshold = np.percentile(distances, 60)
    mnn_pairs = []
    for i in range(df1.shape[0]):
        for j_index in range(num_neighbors_df1):
            j = nearest_neighbors_df1[i, j_index]
            if np.any(nearest_neighbors_df2[j, :] == i) and distances[i, j] < distance_threshold:
                mnn_pairs.append((df1.index[i], df2.index[j]))
    return mnn_pairs
def find_top_half_nearest_neighbors(df1, df2,dis_euclidean):

    distances = cdist(df1.iloc[:, 2:].values, df2.iloc[:, 2:].values, metric='euclidean')
    num_neighbors_df1 = int(dis_euclidean* df2.shape[0])
    nearest_neighbors_df1 = np.argsort(distances, axis=1)[:, :num_neighbors_df1]
    num_neighbors_df2 = int(dis_euclidean* df1.shape[0])
    nearest_neighbors_df2 = np.argsort(distances.T, axis=1)[:, :num_neighbors_df2]
    distance_threshold = np.percentile(distances, 60)
    mnn_pairs = []
    for i in range(df1.shape[0]):
        for j_index in range(num_neighbors_df1):
            j = nearest_neighbors_df1[i, j_index]
            if np.any(nearest_neighbors_df2[j, :] == i) and distances[i, j] < distance_threshold:
                mnn_pairs.append((df1.index[i], df2.index[j]))
    return mnn_pairs

def find_intersection_pairs(pairs_cosine, pairs_euclidean):
    set_pairs_cosine = set(pairs_cosine)
    set_pairs_euclidean = set(pairs_euclidean)
    intersection_pairs = set_pairs_cosine.intersection(set_pairs_euclidean)
    return list(intersection_pairs)

def L2_normalize_rows(df):
    if isinstance(df.iloc[0,2], str):
        data = df.iloc[:, 3:].values
        l2_norm = np.linalg.norm(data, axis=1)
        l2_norm[l2_norm == 0] = 1
        normalized_data = data / l2_norm[:, np.newaxis]
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns[3:])
        normalized_df.insert(0, df.columns[0], df.iloc[:, 0].values)
        normalized_df.insert(1, df.columns[1], df.iloc[:, 1].values)
    else:
        data = df.values
        l2_norm = np.linalg.norm(data, axis=1)
        l2_norm[l2_norm == 0] = 1
        normalized_data = data / l2_norm[:, np.newaxis]
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    return normalized_df
def findMNN_function(save_dir,source_study,target_study,ratio,fraction,dis_cosin,dis_euclidean,CTR_thre,ADA_thre):

    meta_feature= split_data("species","CTR_ADA","Raw_log")
    meta_feature['Sample_ID']=meta_feature.index
    columns_to_move = ['Sample_ID', 'Study', 'Group']
    current_columns = meta_feature.columns.tolist()
    new_order = columns_to_move + [col for col in current_columns if col not in columns_to_move]
    meta_feature = meta_feature[new_order]

    source_df = meta_feature[(meta_feature['Study'].str.contains("|".join(source_study)))]
    target_df= meta_feature[(meta_feature['Study'].str.contains(target_study))]

    source_df_meta= source_df.iloc[:,:3].reset_index(drop=True)
    target_df_meta=target_df.iloc[:,:3].reset_index(drop=True)

    source_norm_df = source_df.drop("Study", axis=1)
    source_norm_df['index'] = source_norm_df["Sample_ID"]
    source_norm_df = source_norm_df.set_index('index')

    target_norm_df=target_df.drop("Study",axis=1)
    target_norm_df['index'] = target_norm_df["Sample_ID"]
    target_norm_df = target_norm_df.set_index('index')

    grouped = target_norm_df.groupby('Group')

    fraction = fraction
    random_target = grouped.apply(lambda x: x.sample(frac=fraction,random_state=42))
    random_target['index'] = random_target['Sample_ID']
    random_target = random_target.set_index('index')

    group_list = random_target["Group"].drop_duplicates().tolist()
    concatenated_filtered_mnn_samples=pd.DataFrame()
    mnn_pairs_finaly = []
    for group in group_list:

        random_subset=random_target[random_target["Group"]==group]
        pairs_euclidean = find_top_half_nearest_neighbors(source_norm_df, random_subset,dis_euclidean)
        pairs_cosine = find_top_half_nearest_cosine_neighbors(source_norm_df, random_subset,dis_cosin)
        top_half_mnn_pairs = find_intersection_pairs(pairs_cosine, pairs_euclidean)

        if group == "ADA":
            thre = ADA_thre
        else:
            thre = CTR_thre
        threshold = int(thre* random_subset.shape[0])

        fr_neighbor_counts = {}
        for i, _ in top_half_mnn_pairs:
            if i in fr_neighbor_counts:
                fr_neighbor_counts[i] += 1
            else:
                fr_neighbor_counts[i] = 1

        frequent_neighbors = [index for index, count in fr_neighbor_counts.items() if count >= threshold]
        filtered_mnn_samples = source_norm_df.loc[frequent_neighbors, :]

        filtered_mnn_samples=filtered_mnn_samples[filtered_mnn_samples["Group"]==group]
        indices = filtered_mnn_samples.index.tolist()
        for i, j in top_half_mnn_pairs:
            if i in indices:
                mnn_pairs_finaly.append((i, j))
        concatenated_filtered_mnn_samples = pd.concat([concatenated_filtered_mnn_samples, filtered_mnn_samples])

    random_select_data = target_norm_df.loc[random_target.index, :]
    remaining_select_data = target_norm_df.drop(random_select_data.index)

    print("random_select_data:", random_select_data.shape[0])
    print("remaining_select_data:", remaining_select_data.shape[0])
    concatenated_final = pd.merge(concatenated_filtered_mnn_samples, source_df_meta[['Sample_ID', 'Study']], on='Sample_ID', how='inner')
    print("target shape:",concatenated_final.shape[0])

    new_order_indices = [0, 1,concatenated_final.shape[1] - 1] + list(range(2, concatenated_final.shape[1] - 1))
    concatenated_filtered = concatenated_final.iloc[:, new_order_indices]
    select_data = pd.merge(random_select_data, target_df_meta[['Sample_ID', 'Study']],
                             on='Sample_ID', how='inner')
    select_data_indices = [0, 1, select_data.shape[1] - 1] + list(range(2, select_data.shape[1] - 1))
    select_data = select_data.iloc[:, select_data_indices]

    remaining_select_data = pd.merge(remaining_select_data, target_df_meta[['Sample_ID', 'Study']],
                              on='Sample_ID', how='inner')
    remaining_select_data_indices = [0, 1, remaining_select_data.shape[1] - 1] + list(range(2, remaining_select_data.shape[1] - 1))
    remaining_select_data = remaining_select_data.iloc[:, remaining_select_data_indices]

    combined_dataframe = pd.concat([concatenated_filtered, select_data], axis=0,ignore_index=True)
    combined_dataframe_all = pd.concat([concatenated_filtered, select_data,remaining_select_data], axis=0, ignore_index=True)
    directory = os.path.dirname(save_dir)
    os.makedirs(directory, exist_ok=True)
    combined_dataframe_all.to_csv(save_dir+'filtered_' + "_".join(source_study) + "_" + target_study + "_"+str(ratio)+'_train_all.csv', index=False)
    combined_dataframe.to_csv(save_dir+'filtered_'+"_".join(source_study)+"_"+target_study+"_"+str(ratio)+'_train.csv', index=False)
    remaining_select_data.to_csv(save_dir+target_study+"_"+str(ratio)+'_test.csv', index=False)
    return select_data,concatenated_filtered,combined_dataframe,remaining_select_data
def read_best_param(filename,study_name):
    data = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            key = row[0]
            value = eval(row[1])
            data[key] = value
    value2 = data[study_name][0]
    return value2['dis_cosin'],value2['dis_euclidean'],value2['CTR_thre'],value2['ADA_thre']
def main(target_study,ratio):
    """
    :param select_data: The data selected by the target data
    :param concatenated_filtered: The data obtained by filtering the source data
    :param combindex_dataframe: The combination of metadata and the filtered data
    :param remaining_select_data: The remaining data
    :return:
    """
    if target_study=="CHN_SH-CRC-4":
        filename = sys.path[1] + "/Result/param/CHN_SH-4_param.csv"
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "CHN_WF-CRC", "CHN_SH-CRC-2", "US-CRC-3"]
    if target_study == "CHN_WF-CRC":
        filename = sys.path[1] + "/Result/param/CHN_WF_param.csv"
        source_study = ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC-2", "US-CRC-3","CHN_SH-CRC-4", "CHN_SH-CRC-2"]

    fraction = float(ratio[1:])
    dis_cosin, dis_euclidean, CTR_thre, ADA_thre = read_best_param(filename, "_".join(
        source_study) + "_" + target_study + "_" + ratio)
    print(dis_cosin, "_", dis_euclidean, "_", CTR_thre, "_", ADA_thre)
    save_dir = data_path + "_".join(source_study) + "_" + target_study + "/"
    random_target, source_studys, train1, remaining_target = findMNN_function(
        save_dir, source_study, target_study,
        ratio=ratio, fraction=fraction,
        dis_cosin=dis_cosin, dis_euclidean=dis_euclidean, CTR_thre=CTR_thre, ADA_thre=ADA_thre)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset selection and matching by a mutual nearest neighbor set algorithm.")
    parser.add_argument("-t", "--target_study", required=True, help="Target study (e.g., CHN_SH-CRC-4 or CHN_WF-CRC).")
    parser.add_argument("-r", "--ratio", required=True, help="S0.4 (e.g., S0.2 or S0.3 or S0.4 or S0.5 or S0.6).")
    args = parser.parse_args()

    target_study = args.target_study
    ratio=args.ratio
    main(target_study,ratio)
    print(f"{target_study} calculation completed.")

