import os
from random import shuffle

orig_dir = '/home/hansencb/Classifcation_Datasets/SkinLesions'
out_dir = 'k_fold_files/'

train_files = []
#val_files = []
# test_files = []

orig_test_class_dirs = os.listdir(os.path.join(orig_dir,'Test'))
orig_train_class_dirs = os.listdir(os.path.join(orig_dir,'Train'))
orig_val_class_dirs = os.listdir(os.path.join(orig_dir,'Validate'))

for c in orig_test_class_dirs:
    img_files = os.listdir(os.path.join(orig_dir, 'Test', c))

    for f in img_files:
        train_files.append((os.path.join(orig_dir, 'Test', c, f), c))

for c in orig_train_class_dirs:
    img_files = os.listdir(os.path.join(orig_dir, 'Train', c))

    for f in img_files:
        train_files.append((os.path.join(orig_dir, 'Train', c, f), c))


for c in orig_val_class_dirs:
    img_files = os.listdir(os.path.join(orig_dir, 'Validate', c))

    for f in img_files:
        train_files.append((os.path.join(orig_dir, 'Validate', c, f), c))


shuffle(train_files)
size = 1000

sets = []

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for i in range(0,len(train_files),size):
    if len(train_files) - i < 0.5 * size:
        k = 0
        for j in range(len(train_files[i:])):
            sets[k].append(train_files[i+j])

            k += 1
            if k == len(sets):
                k = 0
    else:
        sets.append(train_files[i:i+size])

for i in range(0,len(sets)):
    with open(os.path.join(out_dir,'train_fold_{}.txt'.format(i)), 'w') as f:
        for j in range(0,len(sets)):
            if j != i:
                for k in range(0,len(sets[j])):
                    f.write('{} {}\n'.format(sets[j][k][0], sets[j][k][1]))
    with open(os.path.join(out_dir,'test_fold_{}.txt'.format(i)), 'w') as f:
        for k in range(0, len(sets[i])):
            f.write('{} {}\n'.format(sets[i][k][0], sets[i][k][1]))


