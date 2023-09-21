import collections

import torch
import torch.nn as nn
import numpy as np


class RewardShaper:
    def __init__(
        self,
        fn='scale',
        scale=1.0,
    ):
        self.fn = fn
        self.scale = scale

    def __call__(self, rewards):
        if self.fn == 'scale':
            rewards *= self.scale
        else:
            raise NotImplementedError(self.fn)
        return rewards


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer('running_mean', torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer('running_var', torch.ones(in_size, dtype = torch.float64))
        self.register_buffer('count', torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = \
                self._update_mean_var_count_from_moments(
                    self.running_mean, self.running_var, self.count, mean, var, input.size()[0])

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        # get output
        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

    def __repr__(self):
        return f'RunningMeanStd({self.insize})'


class AsyncOutput:
    def __init__(self, callback, parallel=True):
        self._callback = callback
        self._parallel = parallel
        if parallel:
            import concurrent.futures

            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._future = None

    def __call__(self, summaries):
        if self._parallel:
            self._future and self._future.result()
            self._future = self._executor.submit(self._callback, summaries)
        else:
            self._callback(summaries)


class TensorboardLogger(AsyncOutput):
    def __init__(self, tb_dir, config, parallel=True):
        super().__init__(self._write, parallel)
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(tb_dir)

        flat_config = flatten_dict(config)
        try:
            import pandas as pd

            d = pd.DataFrame.from_dict(flat_config, orient='index')
            self.writer.add_text('cfg', d.to_markdown(), 0)
        except ImportError:
            self.writer.add_hparams(flat_config, {})

    def add_video(self, tag, vid_tensor, global_step=None, walltime=None, fps=8):
        from tensorboard.compat.proto.summary_pb2 import Summary

        # from dreamerv3.embodied, since torch.utils.tensorboard.SummaryWriter.add_video is slow (converts to float32)
        def _encode_gif(frames, fps):
            from subprocess import Popen, PIPE
            h, w, c = frames[0].shape
            pxfmt = {1: 'gray', 3: 'rgb24'}[c]
            cmd = ' '.join([
                'ffmpeg -y -f rawvideo -vcodec rawvideo',
                f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
                '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
                f'-r {fps:.02f} -f gif -'])
            proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
            for image in frames:
                proc.stdin.write(image.tobytes())
            out, err = proc.communicate()
            if proc.returncode:
                raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
            del proc
            return out

        def video(tag, tensor, fps):
            if np.issubdtype(tensor.dtype, np.floating):
                tensor = np.clip(255 * tensor, 0, 255).astype(np.uint8)

            tensor = _prepare_video(tensor)
            T, H, W, C = tensor.shape
            tensor_string = _encode_gif(tensor, fps)

            image = Summary.Image(height=H, width=W, colorspace=C, encoded_image_string=tensor_string)
            return Summary(value=[Summary.Value(tag=tag, image=image)])

        self.writer._get_file_writer().add_summary(video(tag, vid_tensor, fps), global_step, walltime)

    def _write(self, summaries):
        for step, name, value in summaries:
            try:
                value = np.asarray(value)

                if len(value.shape) == 0:
                    self.writer.add_scalar(name, value, step)
                elif len(value.shape) == 1:
                    # histogram(name, value, step)
                    raise NotImplementedError
                elif len(value.shape) == 5:
                    self.add_video(name, value, step)
                    # value = value.transpose(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
                    # self.writer.add_video(name, value, step)

            except Exception:
                print('Error writing summary:', name)
                raise
        # self.writer.flush()


class WandbLogger:
    def __init__(self):
        import wandb

        self.wandb = wandb

    def __call__(self, summaries):
        wandb = self.wandb

        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            value = np.asarray(value)
            if len(value.shape) == 0:
                bystep[step][name] = value.item()
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)
            # elif len(value.shape) == 5:
            #     value = value.transpose(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
            #     value = _prepare_video(value)
            #     bystep[step][name] = wandb.Video(value, fps=8)

        for step, metrics in bystep.items():
            wandb.log(metrics, step=step)


# from torch.utils.tensorboard._utils, modified to work with channel-last inputs
def _prepare_video(V):
    # tensor = tensor.transpose(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
    b, t, h, w, c = V.shape

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once
    if not is_power2(V.shape[0]):
        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = V.shape[0] // n_rows

    V = np.reshape(V, newshape=(n_rows, n_cols, t, h, w, c))
    V = np.transpose(V, axes=(2, 0, 3, 1, 4, 5))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))
    return V


# https://stackoverflow.com/a/62186053
def flatten_dict(dictionary, parent_key=False, separator='.'):
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_dict({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)
