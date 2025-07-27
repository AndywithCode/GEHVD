from graphviz import Source
import re
import os

def dot_to_image(dot_file_path, output_image_path):
    # 读取.dot文件内容
    with open(dot_file_path, 'r') as file:
        dot_data = file.read()

    # 使用graphviz将.dot文件转换为图像
    src = Source(dot_data)
    src.render(output_image_path, format='png')  # 可以选择不同格式，如 'pdf', 'svg'

def find_main_dot():
    pattern = r'digraph\s+\"(.*?)\"'
    for root, _, files in os.walk('/home/wyx/VulExplain/test'):
        for file in files:
            if file.endswith('.dot'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                match = re.search(pattern, content)
                if match.group(1) == 'main':
                    return file_path
    return None

if __name__ == "__main__":
    # main_file = find_main_dot()
    dot_file = '/home/wyx/VulExplain/test/269-pdg.dot'  # 你的dot文件路径
    output_image = '/home/wyx/VulExplain/image/269-pdg'  # 输出图像路径（无需扩展名）

    dot_to_image(dot_file, output_image)
    print("图像生成完毕！")
