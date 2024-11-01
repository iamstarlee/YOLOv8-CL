import os

def get_target_classes(classes_path, class_names):

    with open(classes_path, 'r', encoding='utf-8') as f:
        class_list = [line.strip() for line in f.readlines()]
    target_classes = []
    for name in class_names:
        if name in class_list:
            class_id = class_list.index(name)
            target_classes.append(class_id)
        else:
            print(f"Warning: Class '{name}' not found in {classes_path}")
    
    return target_classes


def load_data_with_specific_classes(train_annotation_path, val_annotation_path, target_classes):

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    def filter_lines(lines, target_classes):
        filtered_lines = []
        for line in lines:
            entries = line.strip().split()
            for entry in entries[1:]:
                class_id = int(entry.split(',')[-1])
                if class_id in target_classes:
                    filtered_lines.append(line)
                    break
        return filtered_lines

    train_lines = filter_lines(train_lines, target_classes)
    val_lines = filter_lines(val_lines, target_classes)

    num_train = len(train_lines)
    num_val = len(val_lines)

    return train_lines, val_lines, num_train, num_val
    