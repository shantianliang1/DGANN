import datetime
import time
from tensorboardX import SummaryWriter
from utils import tensor_check
from argSettings import arg

writer = None
scalar, image, audio, text, hist = {}, {}, {}, {}, {}
logged = time.time()


def log(*args):
    print(*args, flush=True)

    if arg.log_file:
        with open(arg.log_dir + arg.log_file, 'a') as f:
            print(*args, flush=True, file=f)


def get_writer():
    global writer
    if writer is None:
        tf_log_dir = arg.log_dir
        tf_log_dir += '' if tf_log_dir.endswith('/') else '/'
        if arg.exp_name:
            tf_log_dir += arg.exp_name
        tf_log_dir += datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')
        writer = SummaryWriter(tf_log_dir)
    return writer


def log_scalar(tag, value, global_step=None):
    scalar[tag] = (tensor_check(value), global_step)


def log_step(epoch=None, global_step=None, max_epoch=None, max_step=None):
    global logged, logged_step, scalar, image, audio, text, hist

    # whether to update logging
    if (arg.log_interval is None and arg.log_step is None) or \
            (arg.log_interval and time.time() - logged >= arg.log_interval) or \
            (arg.log_step and global_step % arg.log_step == 0):

        # update logging
        logged = time.time()
        logged_step = global_step

        console_out = ''
        if epoch:
            console_out += 'ep: %d' % epoch
            if max_epoch:
                console_out += '/%d' % max_epoch
        if global_step:
            if max_step:
                step = global_step % max_step
                step = max_step if step == 0 else step
                console_out += 'step: %d/%d' % (step, max_step)
            else:
                console_out += ' step: %d' % global_step

        # tensorboard
        for k, v in scalar.items():
            get_writer().add_scalar(k, *v)
            # add to console output
            if not k.startswith('weight/') and not k.startswith('gradient/'):
                console_out += ' %s: %f' % (k, v[0])
        for k, v in image.items():
            get_writer().add_image(k, *v)
        for k, v in audio.items():
            get_writer().add_audio(k, *v)
        for k, v in text.items():
            get_writer().add_text(k, *v)
        for k, v in hist.items():
            get_writer().add_histogram(k, *v, 'auto')

        get_writer().file_writer.flush()

        if len(console_out) > 0:
            log(console_out)

        scalar, image, audio, text = {}, {}, {}, {}


