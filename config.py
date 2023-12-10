import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def config():
    parser = argparse.ArgumentParser(description="pythia-model")

    parser.add_argument("--model_name",
                        help="模型名稱",
                        default='pythia-1b-deduped')
    
    parser.add_argument("--train_file_dir",
                        help="訓練集",
                        default='./AICUP_datasets/all-datasets/train_datasets')
    
    parser.add_argument("--trainer_output_file",
                        help="訓練格式路徑",
                        default='./AICUP_datasets/all-datasets/format_file.tsv')
    
    parser.add_argument("--valid_file_dir",
                        help="測試集",
                        default='./AICUP_datasets/opendid_test/opendid_test')
    parser.add_argument("--valid_output_file",
                        help="測試集格式路徑",
                        default='./AICUP_datasets/opendid_test/valid_format_file.tsv')
    
    parser.add_argument("--train_anno_path",
                        help="訓練集標註黨",
                        default='./AICUP_datasets/all-datasets/answer.txt')    

    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="epochs")

    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="batch_size")
    parser.add_argument("--lora",
                        type=str2bool,
                        default=True,
                        help="是否啟用lora")
    parser.add_argument("--wandblog",
                        type=str2bool,
                        default=False,
                        help="是否啟用wandb")

    args = parser.parse_args()

    return args