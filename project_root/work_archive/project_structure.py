import os

SKIP_DIRS = {
    '.venv', 'venv', 'test', 'tests', '.pytest_cache', '.vscode',
    'structure_only.txt', 'struktura.txt', 'project_root',
    'Структура', '__pycache__', 'results_full.txt'
}

SKIP_EXTENSIONS = {'.exe', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.pdf'}

OUTPUT_DIR = 'Структура'
FULL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'struktura.txt')
STRUCTURE_ONLY_FILE = os.path.join(OUTPUT_DIR, 'structure_only.txt')

def write_project_structure(root_dir, full_output_file, structure_only_file, include_content=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(full_output_file, 'w', encoding='utf-8') as full_f, \
         open(structure_only_file, 'w', encoding='utf-8') as structure_f:

        for foldername, subfolders, filenames in os.walk(root_dir):
            subfolders[:] = [d for d in subfolders if d not in SKIP_DIRS]
            relative_path = os.path.relpath(foldername, root_dir)
            indent_level = 0 if relative_path == '.' else relative_path.count(os.sep)
            indent = '    ' * indent_level
            folder_line = f"{indent}[{os.path.basename(foldername)}]\n"

            full_f.write(folder_line)
            structure_f.write(folder_line)

            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                file_ext = os.path.splitext(filename)[1].lower()
                file_line = f"{indent}    {filename}\n"

                full_f.write(file_line)
                structure_f.write(file_line)

                print(f"Обрабатываю файл: {file_path}")

                if include_content and file_ext not in SKIP_EXTENSIONS:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file_content:
                            content = file_content.read()
                            full_f.write(f"{indent}        --- Содержание ---\n")
                            for line in content.splitlines():
                                full_f.write(f"{indent}        {line}\n")
                            full_f.write(f"{indent}        --- Конец ---\n")
                    except Exception as e:
                        full_f.write(f"{indent}        [Ошибка при чтении файла: {e}]\n")
                elif include_content:
                    full_f.write(f"{indent}        [Пропущен: неподдерживаемое расширение]\n")

if __name__ == '__main__':
    write_project_structure('.', FULL_OUTPUT_FILE, STRUCTURE_ONLY_FILE, include_content=True)
