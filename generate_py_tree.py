import os
import sys

def list_py_files(root_dir, prefix=""):
    """递归列出.py文件，生成树状结构"""
    items = os.listdir(root_dir)
    # 过滤掉隐藏文件和目录如.git, __pycache__, wandb等
    items = [item for item in items if not item.startswith('.') and not item.startswith('__pycache__')]
    # 排序：目录在前，文件在后
    dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
    files = [item for item in items if not os.path.isdir(os.path.join(root_dir, item))]
    # 只保留.py文件
    py_files = [f for f in files if f.endswith('.py')]
    
    # 输出当前目录的.py文件
    for i, f in enumerate(py_files):
        is_last = (i == len(py_files) - 1) and (len(dirs) == 0)
        print(prefix + ("└── " if is_last else "├── ") + f)
        new_prefix = prefix + ("    " if is_last else "│   ")
        # 文件没有子项，所以不需要进一步处理
    
    # 递归处理目录
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1)
        print(prefix + ("└── " if is_last else "├── ") + d + "/")
        new_prefix = prefix + ("    " if is_last else "│   ")
        # 递归进入目录
        subdir_path = os.path.join(root_dir, d)
        list_py_files(subdir_path, new_prefix)

if __name__ == "__main__":
    root = "."
    print(".")
    list_py_files(root)
