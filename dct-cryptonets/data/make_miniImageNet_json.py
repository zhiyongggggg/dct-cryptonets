import sys, os
import random
import re
import numpy as np
import argparse
from os import listdir
from os.path import isfile, isdir, join


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Install mini-ImageNet Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, metavar='PATH', help='File-path to directory with datasets')
    return parser.parse_args()


def main(args):
    cwd = os.getcwd()
    dataset_list = ['base', 'val', 'novel']

    cl = -1
    folderlist = []

    datasetmap = {'base': 'train', 'val': 'val', 'novel': 'test'}
    filelists = {'base': {}, 'val': {}, 'novel': {} }
    filelists_flat = {'base': [], 'val': [], 'novel': [] }
    labellists_flat = {'base': [], 'val': [], 'novel': [] }

    # # Find class identities
    # classes = []
    # for root, _, files in os.walk(args.data_dir):
    #     for f in files:
    #         #if f.endswith('.jpg'):
    #         classes.append(f[:-12])
    #
    # classes = list(set(classes))
    """
    for c in classes_arr:
        print(data_path_new +f'{c}')
        os.mkdir(data_path_new +f'{c}')
    
    # Move images to correct location
    for root, _, files in os.walk(data_path):
        for f in tqdm(files, total=600*100):
            if f.endswith('.jpg'):
                class_name = f[:-12]
                image_name = f[-12:]
                src = f'{root}/{f}'
                dst = data_path_new + f'{class_name}/{image_name}'
                shutil.copy(src, dst)
    """
    for dataset in dataset_list:
        with open(datasetmap[dataset] + ".csv", "r") as lines:
            for i, line in enumerate(lines):
                if i == 0:
                    print("i = 0\n")
                    continue
                fid, _, label = re.split(',|\.', line)
                label = label.replace('\n','')
                if not label in filelists[dataset]:
                    folderlist.append(label)
                    filelists[dataset][label] = []
                    fnames = listdir(join(args.data_dir, label))
                    fname_number = [(re.split('_|\.', fname)[0]) for fname in fnames]
                    sorted_fnames = list(zip(*sorted(zip(fnames, fname_number), key=lambda f_tuple: f_tuple[1] )))[0]

                fid = int(fid[-5:])-1
                fname = join(args.data_dir, label, sorted_fnames[i % 600-1])
                filelists[dataset][label].append(fname)
                mark = i
      #  print(mark)

        for key, filelist in filelists[dataset].items():
            cl += 1
            random.shuffle(filelist)
            filelists_flat[dataset] += filelist
            labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()

    for dataset in dataset_list:
        fo = open(args.data_dir + dataset + ".json", "w")
        fo.write('{"label_names": [')
        fo.writelines(['"%s",' % item  for item in folderlist])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_names": [')
        fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write('],')

        fo.write('"image_labels": [')
        fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
        fo.seek(0, os.SEEK_END)
        fo.seek(fo.tell()-1, os.SEEK_SET)
        fo.write(']}')

        fo.close()
        print(f"{dataset} -OK")


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)