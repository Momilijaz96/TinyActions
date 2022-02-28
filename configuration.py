def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'TinyVirat':
        cfg.data_folder = '/home/mijaz/TinyActions/datasets/TinyVIRAT-v2/videos'
        cfg.train_annotations = '/home/mijaz/TinyActions/datasets/TinyVIRAT-v2/tiny_train_v2.json'
        cfg.val_annotations = '/home/mijaz/TinyActions/datasets/TinyVIRAT-v2/tiny_val_v2.json'
        cfg.test_annotations = '/home/mijaz/TinyActions/datasets/TinyVIRAT-v2/tiny_test_v2_public.json'
        cfg.class_map = '/home/mijaz/TinyActions/datasets/TinyVIRAT-v2/class_map.json'
        cfg.num_classes = 26
    elif dataset == 'TinyVirat-d':
        cfg.data_folder = 'datasets/TinyVIRAT-v2/videos'
        cfg.train_annotations = 'datasets/TinyVIRAT-v2/tiny_train_v2.json'
        cfg.val_annotations = 'datasets/TinyVIRAT-v2/tiny_val_v2.json'
        cfg.test_annotations = 'datasets/TinyVIRAT-v2/tiny_test_v2_public.json'
        cfg.class_map = 'datasets/TinyVIRAT-v2/class_map.json'
        cfg.num_classes = 26
    #cfg.saved_models_dir = './results/saved_models'
    #cfg.tf_logs_dir = './results/logs'
    return cfg