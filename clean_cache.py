import os
import os.path
import shutil

from utils.constants import *


def clear_cache_folder(folder_path):
    try:
        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 遍历文件夹中的所有内容
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # 如果是文件，则删除
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # 如果是目录，则递归删除
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"缓存文件夹 '{folder_path}' 已清空。")
        else:
            print(f"文件夹 '{folder_path}' 不存在。")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"清空缓存文件夹时发生错误: {e}")


def main():
    clear_cache_folder(CACHE_DIR)
    # clear_cache_folder(HTML_CACHE_DIR)
    # clear_cache_folder(IMAGE_CACHE_DIR)
    # clear_cache_folder(EXCEL_CACHE_DIR)


if __name__ == "__main__":
    main()
