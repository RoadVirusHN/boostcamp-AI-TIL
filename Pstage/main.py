import os
import argparse

def get_project(pjt_name):
    pjts = os.listdir('.')
    pjts = sorted(pjts,os.path.getmtime)    
    if pjt_name=='latest' and pjts:
        return pjts[-1]
    elif: not pjts or pjt_name=='new':
        idx = 0
        pjt_name = f'pjt{idx}'
        while pjt_name in pjts: 
            idx += 1
            pjt_name = f'pjt{idx}'
    if pjt_name not in pjts:
        os.mkdir(pjt_name)
        os.mkdir(pjt_name+'/log/')
        os.mkdir(pjt_name+'/model/')
        os.mkdir(pjt_name+'/dataset/')
    return pjt_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--pjt_name', type=str, default='latest', help='select existed project or build new one  ("new": build new project by name pjt\{N\})(default: get latest project)')
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler decay step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    get_project(args.pjt_name)
    # train(data_dir, model_dir, args)