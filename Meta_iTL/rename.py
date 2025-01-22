import os


def rename_contents(path):
    """
    递归遍历并重命名 path 及其所有子目录中的文件和目录：
    将名称中含有 CHN_SH-CRC-2 的部分替换为 CHN_SH-CRC-4
    """
    # 列出当前目录下的所有文件和目录
    entries = os.listdir(path)

    for entry in entries:
        old_path = os.path.join(path, entry)

        # 如果是文件，直接改名
        if os.path.isfile(old_path):
            if "CHN_SH-CRC-3" in entry:
                new_entry = entry.replace("CHN_SH-CRC-3", "CHN_SH-CRC-2")
                new_path = os.path.join(path, new_entry)

                os.rename(old_path, new_path)
                print(f"文件重命名：{old_path} -> {new_path}")

        # 如果是目录，先递归处理子目录，再重命名目录本身
        elif os.path.isdir(old_path):
            # 先处理这个子目录里的所有文件和目录
            rename_contents(old_path)

            # 再重命名目录本身
            if "CHN_SH-CRC-3" in entry:
                new_entry = entry.replace("CHN_SH-CRC-3", "CHN_SH-CRC-2")
                new_path = os.path.join(path, new_entry)

                os.rename(old_path, new_path)
                print(f"目录重命名：{old_path} -> {new_path}")


if __name__ == "__main__":
    # 修改为需要处理的根目录
    target_path = "G:/deeplearning/CRC/benchmarker/code/Meta_iTL/Result/"

    rename_contents(target_path)
    print("重命名操作完成！")
