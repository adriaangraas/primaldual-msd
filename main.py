import argparse
import itertools

import numpy as np
import odl
import torch
from dival import get_standard_dataset
from msd_pytorch.msd_model import scaling_module
from torch.utils.data import DataLoader
from primaldual import DualMSDNetFactory, LearnedPrimalDual, \
    PrimalMSDNetFactory

parser = argparse.ArgumentParser(description='Primal Dual stuff')
parser.add_argument('model_type', default='classic')
parser.add_argument('-restore', type=int, default=None)
parser.add_argument('-lr', type=int, default=4)
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

channels_in = 1
channels_out = 1
batch_size = 3
msd_width = 1
msd_depth = 20
msd_dilations = [1, 2, 3, 4, 5]
lr = 10**(-args.lr)

dataset = get_standard_dataset('lodopab')


class UnsqueezingDataset(torch.utils.data.Dataset):
    """Quickfix for squeezing in extra dim in dataset retrieved from
    dival's `get_standard_dataset`.
    """

    def __init__(self, ds):
        self._ds = ds

    def __getitem__(self, item):
        sino = torch.unsqueeze(self._ds[item][0], dim=0)
        recon = torch.unsqueeze(self._ds[item][1], dim=0)
        return (sino, recon)

    def __add__(self, other):
        return self._ds + other

    def __len__(self):
        return len(self._ds)


class NormalizingModule(torch.nn.Module):
    """Wrapper adding pre- and post normalization. Code adjusted from MSDModel
    written by Allard Hendriksen."""

    def __init__(self, other_net, c_in, c_out):
        super().__init__()
        self.scale_in = scaling_module(c_in, c_in)
        self.scale_out = scaling_module(c_out, c_out)
        self.net = torch.nn.Sequential(
            self.scale_in,
            other_net,
            self.scale_out
        )

    @staticmethod
    def dataloader_statistics(dataloader):
        """Normalize input and target data.
        This function goes through all the training data to compute
        the mean and std of the training data.
        It modifies the network so that all future invocations of the
        network first normalize input data and target data to have
        mean zero and a standard deviation of one.
        These modified parameters are not updated after this step and
        are stored in the network, so that they are not lost when the
        network is saved to and loaded from disk.
        Normalizing in this way makes training more stable.
        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:
        """
        mean_in = square_in = mean_out = square_out = 0

        for (data_in, data_out) in dataloader:
            mean_in += data_in.mean()
            mean_out += data_out.mean()
            square_in += data_in.pow(2).mean()
            square_out += data_out.pow(2).mean()

        mean_in /= len(dataloader)
        mean_out /= len(dataloader)
        square_in /= len(dataloader)
        square_out /= len(dataloader)

        std_in = np.sqrt(square_in - mean_in ** 2)
        std_out = np.sqrt(square_out - mean_out ** 2)
        return mean_in, mean_out, std_in, std_out

    def normalize_to(self, dataloader):
        mean_in, mean_out, std_in, std_out = self.dataloader_statistics(
            dataloader)

        # The input data should be roughly normally distributed after
        # passing through scale_in. Note that the input is first
        # scaled and then recentered.
        self.scale_in.weight.data.fill_(1 / std_in)
        self.scale_in.bias.data.fill_(-mean_in / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        self.scale_out.weight.data.fill_(std_out)
        self.scale_out.bias.data.fill_(mean_out)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


train_dataset = dataset.create_torch_dataset('train')
train_dataset = UnsqueezingDataset(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

validation_dataset = dataset.create_torch_dataset('validation')
validation_dataset = UnsqueezingDataset(validation_dataset)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,  # memory error: len(validation_dataset),
    shuffle=False)

test_dataset = dataset.create_torch_dataset('test')
test_dataset = UnsqueezingDataset(test_dataset)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # memory error: len(test_dataset),
    shuffle=False)


def build_ray_trafo():
    """Code taken from Christian Etmann."""

    # ~26cm x 26cm images
    MIN_PT = [-0.13, -0.13]
    MAX_PT = [0.13, 0.13]
    NUM_ANGLES = 1000
    RECO_IM_SHAPE = (362, 362)

    # image shape for simulation
    # IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)
    IM_SHAPE = RECO_IM_SHAPE

    reco_space = odl.uniform_discr(
        min_pt=MIN_PT, max_pt=MAX_PT,
        shape=RECO_IM_SHAPE, dtype=np.float32)
    det_geom = odl.tomo.parallel_beam_geometry(
        reco_space, num_angles=NUM_ANGLES)
    vol_space = odl.uniform_discr(
        min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
        dtype=np.float32)
    geom = odl.tomo.parallel_beam_geometry(
        vol_space, num_angles=NUM_ANGLES,
        det_shape=det_geom.detector.shape
    )
    return odl.tomo.RayTransform(vol_space, geom)


def save(model, optimizer, epoch, path):
    state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    print(f"Saving to {path}")
    torch.save(state, path)


def restore(model, optimizer, path):
    state = torch.load(path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    model.cuda()
    return state["epoch"]


ray_trafo = build_ray_trafo()

model_type = args.model_type
epoch_start = 0

if model_type == 'classic':
    model = LearnedPrimalDual(ray_trafo).to('cuda')
    model = NormalizingModule(
        other_net=model,
        c_in=channels_in,
        c_out=channels_out
    )
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "classic_norm_epoch_{}.torch"
    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"classic_norm_epoch_{args.restore}.torch") + 1
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
elif model_type == 'msd':
    model = LearnedPrimalDual(
        ray_trafo,
        primal_architecture_factory=PrimalMSDNetFactory(
            msd_depth, msd_width,
            msd_dilations),
        dual_architecture_factory=DualMSDNetFactory(
            msd_depth, msd_width,
            msd_dilations),
    )
    model = NormalizingModule(
        other_net=model,
        c_in=channels_in,
        c_out=channels_out
    )
    model = model.to('cuda')
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "msd_norm_epoch_{}.torch"

    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"msd_norm_epoch_{args.restore}.torch") + 1
        # overwrite optim for learning rate, probably losing momentum
        optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
elif model_type == 'msdwithoutnorm':
    model = LearnedPrimalDual(
        ray_trafo,
        primal_architecture_factory=PrimalMSDNetFactory(
            msd_depth, msd_width,
            [1,2,3,4,5,6,7,8]),
        dual_architecture_factory=DualMSDNetFactory(
            msd_depth, msd_width,
            [1,2,3,4,5,6,7,8])
    )
    model = model.to('cuda')
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "msd_epoch_{}.torch"

    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"msd_epoch_{args.restore}.torch") + 1
        # overwrite optim for learning rate, probably losing momentum
        optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
elif model_type == 'msdmore':
    model = LearnedPrimalDual(
        ray_trafo,
        primal_architecture_factory=PrimalMSDNetFactory(
            30, msd_width,
            msd_dilations),
        dual_architecture_factory=DualMSDNetFactory(
            30, msd_width,
            msd_dilations),
    )
    model = NormalizingModule(
        other_net=model,
        c_in=channels_in,
        c_out=channels_out
    )
    model = model.to('cuda')
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "msdmore_norm_epoch_{}.torch"

    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"msdmore_norm_epoch_{args.restore}.torch") + 1
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
elif model_type == 'msdless':
    model = LearnedPrimalDual(
        ray_trafo,
        primal_architecture_factory=PrimalMSDNetFactory(
            10, msd_width,
            msd_dilations),
        dual_architecture_factory=DualMSDNetFactory(
            10, msd_width,
            msd_dilations),
    )
    model = NormalizingModule(
        other_net=model,
        c_in=channels_in,
        c_out=channels_out
    )
    model = model.to('cuda')
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "msdless_norm_epoch_{}.torch"

    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"msdless_norm_epoch_{args.restore}.torch") + 1
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
elif model_type == 'msdlonger':
    model = LearnedPrimalDual(
        ray_trafo,
        n_iter=20,
        n_primal=10,
        n_dual=10,
        primal_architecture_factory=PrimalMSDNetFactory(
            10, msd_width,
            msd_dilations),
        dual_architecture_factory=DualMSDNetFactory(
            10, msd_width,
            msd_dilations),
    )
    model = NormalizingModule(
        other_net=model,
        c_in=channels_in,
        c_out=channels_out
    )
    model = model.to('cuda')
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    model_fname = "msdlonger_norm_epoch_{}.torch"

    if args.restore is not None:
        epoch_start = restore(model, optim,
                              f"msdlonger_norm_epoch_{args.restore}.torch") + 1
    else:
        print("Starting normalization procedure...")
        model.normalize_to(train_loader)
        save(model, optim, epoch_start, model_fname.format(epoch_start))
else:
    raise Exception("Invalid model type.")


def loss_on_loader(loss_fn, loader):
    loss = 0
    with torch.no_grad():
        for (batch_id, sample) in enumerate(loader):
            input, target = sample
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss += loss_fn(output, target)

    return loss / len(loader)


def loss_fn(a, b):
    return torch.nn.functional.mse_loss(a, b)

if not args.test:
    print("Starting training...")
    best_validation_error = np.inf
    validation_error = 0.0
    for epoch in itertools.count(epoch_start):
        print(f"EPOCH: {epoch}")

        val_loss = loss_on_loader(loss_fn, validation_loader)
        print(f"{epoch:05} Validation error: {val_loss: 0.6f}")

        for batch_id, sample in enumerate(train_loader):
            input, target = sample
            input = input.cuda()
            target = target.cuda()

            model.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            print(f"{batch_id}: batch loss {loss}")

            loss.backward()
            optim.step()

        save(model, optim, epoch, model_fname.format(epoch))
else:
    def plot_two(output, target):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2)
        axs[0].set_title("Network output")
        axs[0].imshow(output)
        axs[1].set_title("Target")
        axs[1].imshow(target)
        plt.show()

    def psnr(reco, gt):
        mse = np.mean((np.asarray(reco) - gt)**2)
        if mse == 0.:
            return float('inf')
        data_range = (np.max(gt) - np.min(gt))
        return 20*np.log10(data_range) - 10*np.log10(mse)

    def ssim(reco, gt):
        from skimage.metrics import structural_similarity as ssim
        data_range = (np.max(gt) - np.min(gt))
        return ssim(reco, gt, data_range=data_range)

    print("Starting testing...")
    with torch.no_grad():
        total_psnr = 0
        total_ssim = 0
        amount = 0
        for (batch_id, sample) in enumerate(test_loader):
            input, target = sample
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            output = output.cpu()[0, 0].numpy()
            target = target.cpu()[0, 0].numpy()

            # plot_two(output, target)
            p = psnr(output, target)
            s = ssim(output, target)
            total_psnr += p
            total_ssim += s
            amount += 1
            print(p)
            print(f"Running average, PSNR: {total_psnr / amount}, SSIM: {total_ssim / amount}")

        print(f"Running average, PSNR: {total_psnr / amount}, SSIM: {total_ssim / amount}")
