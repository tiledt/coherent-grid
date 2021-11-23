import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class Mit5kDataset(Pix2pixDataset):
    """Mit 5K images dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        #parser.set_defaults(dataroot=os.path.expanduser('~') + '/data/mit-5k')
        parser.set_defaults(dataroot='./dataset/mit-5k')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(display_freq=2500)
        parser.set_defaults(print_freq=1000)
        parser.set_defaults(save_latest_freq=4500)
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)
        parser.set_defaults(no_instance_edge=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(no_one_hot=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' or opt.phase == 'val' else opt.phase
        load_size = 512 if opt.load_size <= 512 else 1024
        evaluation = "evaluation" if opt.load_size > 1024 else "evaluation-1024"

        if phase != "test":
            input_dir = os.path.join(root, str(load_size), "original")
            input_paths = make_dataset(input_dir, recursive=False, read_cache=True)

            target_dir = os.path.join(root, str(load_size), "C")
            target_paths = make_dataset(target_dir, recursive=False, read_cache=True)

        else:
            input_dir = os.path.join(root, evaluation, "original")
            input_paths = make_dataset(input_dir, recursive=False, read_cache=True)

            target_dir = os.path.join(root, evaluation, "C")
            target_paths = make_dataset(target_dir, recursive=False, read_cache=True)

        instance_paths = []

        return input_paths, target_paths, instance_paths
