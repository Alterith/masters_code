import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA
import time

t0 = time.time()

h = h5py.File('4096_feats_C3D_69_stride_16_val_2.h5py', "r")


data = h['feats'] # it's ok, the dataset is not fetched to memory yet
keys = h['name']

n = data.shape[0] # how many rows we have in the dataset
print(n)
chunk_size = 1000 # how many rows we feed to IPCA at a time, the divisor of n
ipca = IncrementalPCA(n_components=500, batch_size=10)

print("generating ipca matricies")
for i in range(0, n//chunk_size):
    print(i, n//chunk_size)
    ipca.partial_fit(data[i*chunk_size : (i+1)*chunk_size])
print("generating unique keys")
unique_indicies = []
key_counts = []
unique_keys = []
curr_key = ""
curr_count = 0
for i, name in enumerate(keys):
    
    if name != curr_key:
        unique_keys.append(name[0])

        unique_indicies.append(i)

        curr_key = name
        # case for first element
        if i != 0:
            key_counts.append(curr_count)
            curr_count = 0
    # case for end of list
    curr_count += 1
    if i == (n-1):
        key_counts.append(curr_count)

key_counts = np.asarray(key_counts)
unique_keys, unique_indicies = np.unique(unique_keys, return_index=True)
key_counts = key_counts[unique_indicies]
# takes too much ram so I did the above
# unique_keys, key_counts = np.unique(keys, return_counts=True)
count_idx = np.zeros(key_counts.shape)
count_idx[0] = int(key_counts[0] - 1)

for i in range(1, count_idx.shape[0]):
    count_idx[i] = int(count_idx[i-1] + key_counts[i])

count_idx = count_idx.astype("int32")
print(unique_keys[0])
j = 0
# specify append as file open arg
final_feat_file = h5py.File('500_feats_C3D_69_stride_16_val_2.hdf5', "a")
print("applying pca to data")
for i in range(0, len(key_counts)):
    print(i)
    print(j, (count_idx[i] + 1))
    print(unique_keys[i])
    feats = ipca.transform(data[j:(count_idx[i] + 1)])

    j = count_idx[i] + 1
    name = unique_keys[i].decode('UTF-8')
    name = "v_" + name
    print(name, name[0])
    feats_dict = {
        name: feats,
    }

    for k, v in feats_dict.items():
        final_feat_file.create_dataset(k, data=v)

t1 = time.time()

total = t1-t0

print(total)
