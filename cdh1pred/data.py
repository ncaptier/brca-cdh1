from torch.utils.data import Dataset
import torch
import h5py
from stamp.modeling.marugoto.transformer.data import MapDataset, EncodedDataset


class BagDistDataset(Dataset):

    def __init__(self, bags, bag_size):
        self.bags = bags
        self.bag_size = bag_size

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int):
        # collect all the features
        feats = []
        coords = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, 'r') as f:
                feats.append(torch.from_numpy(f['feats'][:]))
                coords.append(torch.from_numpy(f['coords'][:]))
        feats = torch.concat(feats).float()
        coords = torch.concat(coords).float()

        # sample a subset, if required
        if self.bag_size:
            return _to_fixed_size_bag(feats, coords, bag_size=self.bag_size)
        else:
            return feats, coords, len(feats)


def _to_fixed_size_bag(bag, coords, bag_size: int = 512):
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]
    coord_samples = coords[bag_idxs]

    # zero-pad if we don't have enough samples
    temp = bag_size - bag_samples.shape[0]
    zero_padded = torch.cat((bag_samples,
                             torch.zeros(temp, bag_samples.shape[1])))

    zero_padded_coord = torch.cat((coord_samples,
                                   torch.zeros(temp, coord_samples.shape[1])))

    return zero_padded, zero_padded_coord, min(bag_size, len(bag))


def make_dist_dataset(
        bags,
        target_enc,
        targs,
        bag_size=None,
):
    assert len(bags) == len(targs), \
        'number of bags and ground truths does not match!'

    ds = MapDataset(
        zip_bag_targ,
        BagDistDataset(bags, bag_size=bag_size),
        EncodedDataset(target_enc, targs),
    )

    return ds


def zip_bag_targ(bag, targets):
    features, coords, lengths = bag
    return (
        features,
        coords,
        lengths,
        targets.squeeze(),
    )
