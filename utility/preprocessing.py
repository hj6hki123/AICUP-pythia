import os
from pathlib import Path

def save_content_to_file(file_path, content_list):
    with open(file_path, 'w', encoding='utf-8') as file_writer:
        file_writer.writelines(content_list)

def load_lines_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file_reader:
        return file_reader.readlines()

def extract_answers_from_file(file_path) -> dict:
    answers = {}
    file_lines = load_lines_from_file(file_path)
    for line in file_lines:
        parts = line.rstrip('\n').split('\t')
        data = {
            'phi': parts[1],
            'start_index': int(parts[2]),
            'end_index': int(parts[3]),
            'content': parts[4]
        }
        if len(parts) > 5: 
            data['normalized'] = parts[5]
        answers.setdefault(parts[0], []).append(data)
    return answers

def format_data(training_data_path, answers_file_path, output_path):
    print('Processing Starting!')
    
    answers = extract_answers_from_file(answers_file_path)
    formatted_sequences = []
    for document_name in answers.keys():
        document_path = Path(training_data_path) / f"{document_name}.txt"
        document_content = "".join(load_lines_from_file(document_path))

        start_idx, annotation_idx = 0, 0
        sequence = ""
        formatted_pairs = []
        for idx, char in enumerate(document_content):
            if char == '\n':
                end_line_idx = idx + 1
                if document_content[start_idx:end_line_idx] == '\n':
                    continue
                if not sequence:
                    sequence = "PHI:Null"
                line = document_content[start_idx:end_line_idx].strip().replace('\t', ' ')
                sequence = sequence.rstrip('\\n')
                start_idx = end_line_idx
                formatted_pairs.append(f"{document_name}\t{end_line_idx}\t{line}\t{sequence}\n")
                sequence = ""

            if idx == answers[document_name][annotation_idx]['start_index']:
                annotation = answers[document_name][annotation_idx]
                sequence += f"{annotation['phi']}:{annotation['content']}"
                if 'normalized' in annotation:
                    sequence += f"=>{annotation['normalized']}\\n"
                else:
                    sequence += "\\n"
                if annotation_idx < len(answers[document_name]) - 1:
                    annotation_idx += 1
        formatted_sequences.extend(formatted_pairs)

    save_content_to_file(output_path, formatted_sequences)
    print('Processing Complete!')

if __name__ == '__main__':
    format_data(r'AICUP_datasets/all-datasets/train_datasets',
                r'AICUP_datasets/all-datasets/answer.txt',
                r'AICUP_datasets/all-datasets/format_file.tsv')
