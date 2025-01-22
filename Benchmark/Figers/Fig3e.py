import os
import sys
import pandas as pd
from ete3 import Tree, TreeNode

# Update the system path to include the project directory
new_path = os.path.abspath(os.path.join(__file__, "../../"))
sys.path[1] = new_path

# Define paths for figures, features, and data
figure_path = sys.path[1] + '\\Result\\figures\\Fig03\\'
feature_path = sys.path[1] + '\\Result\\feature\\'
data_path = sys.path[1] + '\\Result\\data\\'


def pre_sig_feature(feature_type, data_type, group):
    """
    Extract significant species and their corresponding taxonomy (Kingdom, Phylum, Class, Order, Family, Genus, Species).

    :param feature_type: Type of feature to process (e.g., species, genus, etc.)
    :param data_type: Type of data being processed (e.g., Raw, Processed, etc.)
    :param group: Experimental group (e.g., CTR_CRC, etc.)
    """
    # Read taxonomy data from the specified file
    text_file_path = data_path + '/mpa_vOct22_CHOCOPhlAnSGB_202212_species.txt'
    data = pd.read_csv(text_file_path, sep="\t", header=None, names=["t_sgb", "Taxonomy"])

    # Split taxonomy into columns
    taxonomy_data = pd.DataFrame()
    taxonomy_data['Taxonomy'] = data['Taxonomy'].str.split(',', expand=True).iloc[:, 0]
    taxonomy_split = taxonomy_data['Taxonomy'].str.split('|', expand=True)
    taxonomy_split.columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    taxonomy_split['Species'] = taxonomy_split['Species'].str.split('__', expand=True).iloc[:, 1]

    # Load feature data
    new_file_path = f"{feature_path}/{feature_type}/{data_type}/{group}/feature.csv"
    new_data = pd.read_csv(new_file_path)

    # Define prefixes for feature types
    prefixes = {
        'Class': 'c__',
        'Order': 'o__',
        'Family': 'f__',
        'Genus': 'g__',
        'Species': '',
        "t_sgb": ''
    }
    prefix = prefixes.get(feature_type, '')
    # Add prefix to species IDs based on the group
    new_data_df = pd.DataFrame()
    if group == "CTR_CRC":
        new_data_df['species_ID'] = prefix + new_data[
            'FR-CRC_AT-CRC_ITA-CRC_JPN-CRC_CHN_WF-CRC_CHN_SH-CRC_CHN_HK-CRC_DE-CRC_IND-CRC_US-CRC_rf_optimal']
    else:
        new_data_df["species_ID"] = prefix + new_data[
            'AT-CRC_JPN-CRC_FR-CRC_CHN_WF-CRC_ITA-CRC_CHN_SH-CRC-2_US-CRC-3_CHN_SH-CRC-4_US-CRC-2_rf_optimal']
    # Merge feature data with taxonomy data
    merged_data = pd.merge(new_data_df, taxonomy_split, left_on='species_ID', right_on="Species", how='inner')
    deduplicated_data = merged_data.drop_duplicates(subset='Species', keep='first')
    # Save the processed data
    new_output_path = f"{figure_path}{feature_type}/{data_type}/{group}/All_features_tools_tree.csv"
    os.makedirs(os.path.dirname(new_output_path), exist_ok=True)
    deduplicated_data.to_csv(new_output_path, index=False)
    print(f"File saved to: {new_output_path}")


def tree_function():
    """
    Generate a tree file based on taxonomic hierarchy.
    """
    def build_tree(data, hierarchy):
        """
        Build a tree structure from the taxonomic hierarchy.

        :param data: DataFrame containing taxonomy data
        :param hierarchy: List of taxonomic levels in hierarchical order
        :return: Root node of the tree
        """
        root = TreeNode(name="Root")
        for _, row in data.iterrows():
            current_node = root
            for level in hierarchy:
                taxon_name = row[level]
                found = False
                for child in current_node.children:
                    if child.name == taxon_name:
                        current_node = child
                        found = True
                        break
                if not found:
                    new_node = TreeNode(name=taxon_name)
                    current_node.add_child(new_node)
                    current_node = new_node
        return root

    # Load data and define hierarchy
    data = pd.read_csv(f"{figure_path}{feature_type}/{data_type}/{group}/All_features_tools_tree.csv")
    hierarchy = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Build and save the tree
    tree_root = build_tree(data, hierarchy)
    print(tree_root.get_ascii(show_internal=True))
    tree_root.write(outfile=f"{figure_path}{feature_type}/{data_type}/{group}/All_features_tools_tree.nwk")


def merge_fold(feature_type,data_type, group):
    """
    Merge fold-change data from multiple studies and perform statistical analysis.

    :param data_type: Type of data being processed (e.g., Raw, Processed, etc.)
    :param group: Experimental group (e.g., CTR_CRC, etc.)
    """
    # Load the primary file
    file1_path = f"{figure_path}{feature_type}/{data_type}/{group}/All_features_tools_tree.csv"
    df1 = pd.read_csv(file1_path)

    # Define study groups
    studies = (
        ["FR-CRC", "AT-CRC", "ITA-CRC", "JPN-CRC", "US-CRC", "IND-CRC", "CHN_WF-CRC", "CHN_SH-CRC", "CHN_HK-CRC",
         "DE-CRC"]
        if group == "CTR_CRC" else
        ['AT-CRC', 'JPN-CRC', 'FR-CRC', 'CHN_WF-CRC', 'ITA-CRC', "CHN_SH-CRC-4", "CHN_SH-CRC-2", "US-CRC-2", "US-CRC-3"]
    )

    # Initialize the merged DataFrame
    final_merged_df = df1.copy()

    # Merge data from each study
    for study in studies:
        file2_path = f"{figure_path}{feature_type}/{data_type}/{group}/{study}_FC_feature_study.csv"
        df2 = pd.read_csv(file2_path)

        merged_df = pd.merge(final_merged_df, df2[['Species', 'FC_Value']], left_on='species_ID', right_on='Species',
                             how='left')
        merged_df.rename(columns={'FC_Value': study}, inplace=True)
        merged_df.drop(columns='Species_y', inplace=True)
        merged_df.rename(columns={'Species_x': 'Species'}, inplace=True)

        final_merged_df = merged_df

    # Save the merged data
    output_path = f"{figure_path}{feature_type}/{data_type}/{group}/fold_change.csv"
    final_merged_df.to_csv(output_path, index=False)
    print(f"Merged file saved to {output_path}")

    # Perform statistical calculations
    columns_to_check = studies
    final_merged_df['count_equal_zero'] = final_merged_df[columns_to_check].apply(lambda x: (x == 0).sum(), axis=1)
    final_merged_df['count_less_than_zero'] = final_merged_df[columns_to_check].apply(lambda x: (x < 0).sum(), axis=1)
    final_merged_df['count_greater_than_zero'] = final_merged_df[columns_to_check].apply(lambda x: (x > 0).sum(),
                                                                                         axis=1)
    final_merged_df['count>0'] = final_merged_df[columns_to_check].apply(
        lambda x: (x > 0).sum() / len(columns_to_check), axis=1)
    final_merged_df['count<0'] = final_merged_df[columns_to_check].apply(
        lambda x: (x < 0).sum() / len(columns_to_check), axis=1)
    final_merged_df['count1'] = final_merged_df.apply(lambda row: max(row['count>0'], row['count<0']) * 100, axis=1)

    # Save the updated DataFrame
    final_merged_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    feature_type = "species"
    group = "CTR_ADA"
    data_type = "Raw_log"

    # Uncomment the function calls as needed
    # pre_sig_feature(feature_type,data_type, group)
    # tree_function()
    merge_fold(feature_type,data_type, group)
