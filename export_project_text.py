import os

# 可导出的文件类型
EXPORT_EXTS = (".py")
# 忽略的文件夹
IGNORE_DIRS = {"venv"}

OUTPUT_FILE = "project_export.txt"

def is_code_file(filename):
    return filename.endswith(EXPORT_EXTS)

def should_ignore_dir(dirname):
    return any(ignore in dirname for ignore in IGNORE_DIRS)

def format_title(text, level=1):
    """生成章节标题"""
    prefix = "#" * level
    return f"\n\n{prefix} {text}\n{'=' * len(text)}\n"

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("项目代码文本汇总\n")
        out.write("====================\n")
        out.write("（自动生成，包含所有源码文件内容）\n\n")

        # 遍历目录树
        for root, dirs, files in os.walk("."):
            if should_ignore_dir(root):
                continue

            # 生成章节名
            folder_name = os.path.relpath(root, ".")
            if folder_name == ".":
                folder_name = "项目根目录"
            out.write(format_title(folder_name, level=2))

            # 逐文件写入
            for file in sorted(files):
                if not is_code_file(file):
                    continue

                file_path = os.path.join(root, file)
                out.write(format_title(file, level=3))

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    content = "[无法读取：非UTF-8编码文件]\n"

                out.write(content)
                out.write("\n\n" + "-" * 80 + "\n")

    print(f"✅ 已生成结构化文本文件：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
