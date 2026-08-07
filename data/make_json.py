import os
import json


PATH = 'xxx/MPDD'
SAVE_DIR = 'MPDD'

if __name__ == '__main__':
    cls_name_list = os.listdir(PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_json_f = open(os.path.join(SAVE_DIR, 'train.json'), 'w', encoding='utf-8')
    for cls_name in cls_name_list:
        train_rel_path = os.path.join(cls_name, 'train')
        train_abs_path = os.path.join(PATH, train_rel_path)
        defect_name_list = os.listdir(train_abs_path)
        for defect_name in defect_name_list:
            train_def_rel_path = os.path.join(train_rel_path, defect_name)
            train_def_abs_path = os.path.join(PATH, train_def_rel_path)
            img_name_list = os.listdir(train_def_abs_path)
            for img_name in img_name_list:
                json_dict = {
                    'filename': os.path.join(train_def_rel_path, img_name),
                    'label': 0,
                    'label_name': defect_name,
                    'clsname': cls_name,
                }
                json_str = json.dumps(json_dict)
                train_json_f.write(json_str + '\n')
    train_json_f.close()

    test_json_f = open(os.path.join(SAVE_DIR, 'test.json'), 'w', encoding='utf-8')
    for cls_name in cls_name_list:
        gt_rel_path = os.path.join(cls_name, 'ground_truth')
        test_rel_path = os.path.join(cls_name, 'test')
        test_abs_path = os.path.join(PATH, test_rel_path)
        defect_name_list = os.listdir(test_abs_path)
        for defect_name in defect_name_list:
            gt_def_rel_path = os.path.join(gt_rel_path, defect_name)
            test_def_rel_path = os.path.join(test_rel_path, defect_name)
            test_def_abs_path = os.path.join(PATH, test_def_rel_path)
            img_name_list = os.listdir(test_def_abs_path)
            for img_name in img_name_list:
                if defect_name == 'good':
                    json_dict = {
                        'filename': os.path.join(test_def_rel_path, img_name),
                        'label': 0,
                        'label_name': defect_name,
                        'clsname': cls_name,
                    }
                else:
                    json_dict = {
                        'filename': os.path.join(test_def_rel_path, img_name),
                        'label': 1,
                        'label_name': defect_name,
                        'maskname': os.path.join(gt_def_rel_path, img_name.replace('.png', '_mask.png')),
                        'clsname': cls_name,
                    }
                json_str = json.dumps(json_dict)
                test_json_f.write(json_str + '\n')
    test_json_f.close()