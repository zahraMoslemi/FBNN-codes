from src.base_net import *
from .optimizers import *
import torch.nn as nn
import torch.nn.functional as F
import copy


class MLP(nn.Module):
    def __init__(self, input_dim, width1, width2, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width1 = width1
        self.width2 = width2

        layers = [
          nn.Linear(input_dim, width1), 
          nn.ReLU(), 
          nn.Linear(width1, width2),
          nn.ReLU(),   
          nn.Linear(width2, output_dim)
          ]
        # 3,5,5,1

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BNN_cat(BaseNet):  # for categorical distributions
    def __init__(self, N_train, lr=1e-2, cuda=True, grad_std_mul=30):
        super(BNN_cat, self).__init__()

        cprint('y', 'BNN categorical output')
        self.lr = lr
        self.model = MLP(input_dim=14, width1=60, width2=30,output_dim=1)
        self.cuda = cuda

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

        self.grad_buff = []
        self.max_grad = 1e20
        self.grad_std_mul = grad_std_mul

        self.weight_set_samples = []

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)
        if self.cuda:
            self.model.cuda()

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        """This optimiser incorporates the gaussian prior term automatically. The prior variance is gibbs sampled from
        its posterior using a gamma hyper-prior."""
        self.optimizer = H_SA_SGHMC(params=self.model.parameters(), lr=self.lr, base_C=0.05, gauss_sig=0.1)  # this last parameter does nothing

    def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
        self.set_mode_train(train=True)
        x, y = to_variable(var=(x, y.float()), cuda=self.cuda)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out, y, reduction='mean')
        loss = loss * self.N_train  # We use mean because we treat as an estimation of whole dataset
        loss.backward()

        # Gradient buffer to allow for dynamic clipping and prevent explosions
        if len(self.grad_buff) > 1000:
            self.max_grad = np.mean(self.grad_buff) + self.grad_std_mul * np.std(self.grad_buff)
            self.grad_buff.pop(0)
        # Clipping to prevent explosions
        self.grad_buff.append(nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                       max_norm=self.max_grad, norm_type=2))
        if self.grad_buff[-1] >= self.max_grad:
            print(self.max_grad, self.grad_buff[-1])
            self.grad_buff.pop()
        self.optimizer.step(burn_in=burn_in, resample_momentum=resample_momentum, resample_prior=resample_prior)

        # out: (batch_size, out_channels, out_caps_dims)
        # pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        probs = F.sigmoid(out).data.cpu()
        pred = (probs>.5).float()
        err = pred.ne(y.data).sum()

        return loss.data * x.shape[0] / self.N_train, err

    def eval(self, x, y, train=False):
        self.set_mode_train(train=False)
        x, y = to_variable(var=(x, y.float()), cuda=self.cuda)

        out = self.model(x)
        loss = F.binary_cross_entropy_with_logits(out, y, reduction='sum')
        probs = F.sigmoid(out).data.cpu()

        # pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        pred = (probs>.5).float()
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def save_sampled_net(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
        return None

    def predict(self, x):
        self.set_mode_train(train=False)
        x, = to_variable(var=(x, ), cuda=self.cuda)
        out = self.model(x)
        probs = F.sigmoid(out).data.cpu()
        return probs.data

    def sample_predict(self, x, Nsamples=0, grad=False):
        """return predictions using multiple samples from posterior"""
        self.set_mode_train(train=False)
        if Nsamples == 0:
            Nsamples = len(self.weight_set_samples)
        x, = to_variable(var=(x, ), cuda=self.cuda)

        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True

        out = x.data.new(Nsamples, x.shape[0], self.model.output_dim)

        # iterate over all saved weight configuration samples
        for idx, weight_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break
            self.model.load_state_dict(weight_dict)
            out[idx] = self.model(x)

        out = out[:idx]
        prob_out = F.softmax(out, dim=2)

        if grad:
            return prob_out
        else:
            return prob_out.data

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        for idx, state_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break

            for key in state_dict.keys():
                if 'weight' in key:
                    weight_mtx = state_dict[key].cpu().data
                    for weight in weight_mtx.view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def save_weights(self, filename):
        save_object(self.weight_set_samples, filename)

    def load_weights(self, filename, subsample=1):
        self.weight_set_samples = load_object(filename)
        self.weight_set_samples = self.weight_set_samples[::subsample]