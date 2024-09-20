import os
import numpy as np
import time
import logging
import sys
from memory_profiler import profile
from concrete import fhe



# 1 filter, 1st block kernel (2,2) stride 2 pad 0, 2nd block kernel (2,2) stride 1 pad 0
# LR 4 bits
# 4 cores Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz --> 192.7s    + cuda on device
# 48 cores Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz --> 123.8s   + cuda on device
# 8 cores Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz --> 4.9s



def bytes_to_gb(bytes_val):
    gb = bytes_val / (1024 ** 3)  # 1 GB = 1024^3 bytes
    return gb


def bytes_to_mb(bytes_val):
    mb = bytes_val / (1024 ** 2)  # 1 MB = 1024^2 bytes
    return mb


def bytes_to_kb(bytes_val):
    kb = bytes_val / (1024 ** 1)  # 1 MB = 1024^2 bytes
    return kb


def bytes_to_xb(bytes_val):
    gb = bytes_to_gb(bytes_val)
    if int(gb) == 0:
        mb = bytes_to_mb(bytes_val)
        if int(mb) == 0:
            kb = bytes_to_kb(bytes_val)
            if int(kb) == 0:
                return f'{round(bytes_val, 2)} bytes'
            else:
                return f'{round(kb, 2)} kB'
        else:
            return f'{round(mb, 2)} MB'
    else:
        return f'{round(gb, 2)} GB'


def pos_conv2D(shape, ksize=(3, 3), pad=0, stride=2):
    """Return the indices used in the 2D convolution"""

    arr = np.array(list(range(shape[0] * shape[1]))).reshape((shape))
    height, width = arr.shape

    kernel_height, kernel_width = ksize

    output_height = int((height + 2 * pad - kernel_height) / stride) + 1
    output_width = int((width + 2 * pad - kernel_width) / stride) + 1

    out = np.zeros((output_height, output_width, kernel_height * kernel_width))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride
            h_end = h_start + kernel_height
            w_start = w * stride
            w_end = w_start + kernel_width
            # Get the receptive_field
            # pad image # shape (B, 8, 10, 10)
            receptive_field = arr[h_start:h_end, w_start:w_end].reshape(
                (kernel_height * kernel_width))  # shape(B, 6) binaire
            # transform input 0/1 into int between [0 ; 2**n-1]
            out[h, w, :] = receptive_field  # output_var_unfold

    return out.astype(int)


def conv2D_unfold(image, ksize=(3, 3), pad=0, stride=2):
    """Return the indices used in the 2D convolution"""

    nfilters, height, width = image.shape
    # arr = np.array(list(range(height * width))).reshape((image.shape))

    kernel_height, kernel_width = ksize

    output_height = int((height + 2 * pad - kernel_height) / stride) + 1
    output_width = int((width + 2 * pad - kernel_width) / stride) + 1

    out = fhe.zeros((nfilters, output_height, output_width, kernel_height * kernel_width))

    for f in range(nfilters):
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * stride
                h_end = h_start + kernel_height
                w_start = w * stride
                w_end = w_start + kernel_width
                # Get the receptive_field
                # pad image # shape (B, 8, 10, 10)
                receptive_field = image[f, h_start:h_end, w_start:w_end].reshape(
                    (kernel_height * kernel_width))  # shape(B, 6) binaire
                # transform input 0/1 into int between [0 ; 2**n-1]
                out[f,h, w, :] = receptive_field  # output_var_unfold

    return out.astype(int)


def conv2D_unfold_deep(image, ratio, ksize=(1, 1), pad=0, stride=1):
    """Return the indices used in the 2D convolution"""

    nfilters, height, width = image.shape
    # arr = np.array(list(range(height * width))).reshape((image.shape))

    kernel_height, kernel_width = ksize

    output_height = int((height + 2 * pad - kernel_height) / stride) + 1
    output_width = int((width + 2 * pad - kernel_width) / stride) + 1

    out = fhe.zeros((nfilters//ratio, output_height, output_width, kernel_height * kernel_width*ratio))

    for f in range(0,nfilters,ratio):
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * stride
                h_end = h_start + kernel_height
                w_start = w * stride
                w_end = w_start + kernel_width
                # Get the receptive_field
                # pad image # shape (B, 8, 10, 10)
                receptive_field = image[f:f+ratio, h_start:h_end, w_start:w_end].reshape(
                    (kernel_height * kernel_width * ratio))  # shape(B, 6) binaire
                # transform input 0/1 into int between [0 ; 2**n-1]
                out[f//ratio, h, w, :] = receptive_field  # output_var_unfold

    return out.astype(int)


def BitsToIntAFast(bits):

    if len(bits.shape) == 3:
        _, m, n = bits.shape  # number of columns is needed, not bits.size
    else:
        m, n = bits.shape[-2:]
    a = 2 ** np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits

    return (bits @ a).astype(int)  # this matmult is the key line of code


def available_cores():
    # Get the CPU affinity mask for the current process
    affinity_mask = os.sched_getaffinity(0)
    num_cores_available = len(affinity_mask)

    # print("CPU affinity mask:", affinity_mask)
    # print("Number of available CPU cores:", num_cores_available)

    return affinity_mask, num_cores_available


def create_logger(base_path, logger_name='my_logger', log_dir='./logs'):
    affinity_mask, num_cores_available = available_cores()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    original_umask = os.umask(0)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, 0o777)
    if not os.path.exists(os.path.join(log_dir, base_path)):
        os.makedirs(os.path.join(log_dir, base_path),0o777)

    file_handler = logging.FileHandler(os.path.join(log_dir, base_path, f'{base_path}_{num_cores_available}_cores.log'),
                                       mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"CPU affinity mask: {affinity_mask}")
    logger.info(f"Number of available CPU cores: {num_cores_available}")

    return logger


def run_one_input(circuit, inpt, logger, base_path, log_dir='./logs'):
    enc = circuit.encrypt(*inpt)

    logger.info(f"Encryption done")

    affinity_mask, num_cores_available = available_cores()

    @profile(stream=open(os.path.join(log_dir, base_path, f'{base_path}_{num_cores_available}_cores.log'), "a"))
    def run(x, log):

        t = time.time()
        try:
            out = circuit.run(x)
        except Exception as e:
            log.error(e, exc_info=True)
            sys.exit(3)
        t_run = time.time() - t

        return out, t_run

    res, t_final = run(enc, logger)

    logger.info(f"Time run: {t_final} seconds")

    return circuit, res, t_final


def generate_keys(circuit, base_path, logger, key_dir='./keys'):
    key_dir = os.path.join(key_dir, base_path)
    original_umask = os.umask(0)
    if not os.path.exists(key_dir):
        os.makedirs(key_dir, 0o777)

    t = time.time()
    circuit.keys.load_if_exists_generate_and_save_otherwise(os.path.join(key_dir, "keys"))
    # circuit.keygen()
    t_keygen = time.time() - t
    logger.info(f"Keygen done in {t_keygen}")

    logger.info(f"Encryption keys: {bytes_to_xb(circuit.size_of_secret_keys)}")
    logger.info(f"Evaluation keys: {bytes_to_xb(circuit.size_of_bootstrap_keys + circuit.size_of_keyswitch_keys)}")
    logger.info(f"Inputs: {bytes_to_xb(circuit.size_of_inputs)}")
    logger.info(f"Outputs: {bytes_to_xb(circuit.size_of_outputs)}")

    return circuit


def compile_circuit(func, inputset, logger, configuration, multi_precision=True):
    # logger.debug(f'inputset shape : {inputset.shape}')

    logger.info(f"Compilation ...")

    t = time.time()
    try:
        if multi_precision:
            circuit = func.compile(inputset, configuration,
                                   single_precision=not (multi_precision),
                                   parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI,
                                   )
        else:
            circuit = func.compile(inputset, configuration,
                                   parameter_selection_strategy=fhe.ParameterSelectionStrategy.MONO)
    except Exception as e:
        logger.error(e, exc_info=True)
        sys.exit(3)

    logger.info(f"Compilation done in {time.time() - t} seconds")

    return circuit


def compile_keygen_run(func, inputset, sample, base_path, logger, configuration, multi_precision=True,
                       log_dir='./logs', key_dir='./keys'):
    affinity_mask, num_cores_available = available_cores()

    @profile(stream=open(os.path.join(log_dir, base_path, f'{base_path}_{num_cores_available}_cores.log'), "a"))
    def c():
        circuit = compile_circuit(func, inputset, logger, configuration, multi_precision)
        return circuit

    circuit = c()

    logger.info('Compilation done')

    @profile(stream=open(os.path.join(log_dir, base_path, f'{base_path}_{num_cores_available}_cores.log'), "a"))
    def k(my_circuit):
        circuit = generate_keys(my_circuit, base_path, logger, key_dir)
        return circuit

    circuit = k(circuit)

    logger.info('Key generation done')

    circuit, res, t_final = run_one_input(circuit, sample, logger, base_path, log_dir)

    logger.info('Run done')

    return circuit, res, t_final

nfilters1, nfilters2 = 1, 1 # 16,16

kw1, kh1 = 2, 2
kw2, kh2 = 2, 2
nbits1, stride1, pad1 = kw1*kh1, 2, 0
nbits2, stride2, pad2 = kw2*kh2, 1, 0

block1 = np.random.randint(0,2, (2**nbits1, nfilters1)).astype(int)
block2 = np.random.randint(0,2, (2**nbits2, nfilters2)).astype(int)


indexes_block1 = pos_conv2D((20,20), ksize=(kw1,kh1),stride=stride1,pad=pad1)
width_input_block2, height_input_block2, _ = indexes_block1.shape
indexes_block2 = pos_conv2D((width_input_block2,height_input_block2), ksize=(kw2,kh2),stride=stride2,pad=pad2)
s1,s2, _ = indexes_block2.shape


# {2: 0.0974, 3: 0.0974, 4: 0.0974, 5: 0.2482, 6: 0.0974, 7: 0.6514, 8: 0.9513, 9: 0.9713, 10: 0.972, 11: 0.977, 12: 0.9758, 13: 0.976, 14: 0.9761, 15: 0.9763, 16: 0.9764}
nbitslr = 4
# w_quant, b_quant = quant_weight_bias(W,B, nbitslr)
# w_quant = w_quant.transpose()
w_quant = np.random.randint(0,2**nbitslr, (10, s1*s2*nfilters2)).astype(int)# classifier_bn_weight.reshape(-1,1) * linear_weight / classifier_bn_std.reshape(-1,1)
b_quant = np.random.randint(0,2**nbitslr, (10)).astype(int) # classifier_bn_bias + classifier_bn_weight * linear_bias / classifier_bn_std - classifier_bn_weight * classifier_bn_mean / classifier_bn_std
nclasses, nfeatures = w_quant.shape
N = 18  # 18 12 = number of subsums
end = nfeatures // N  # 32 48 = nfeatures // N

w_bits = [((w_quant >> i) & 1).astype(int).transpose() for i in range(nbitslr)]


npatches1 = indexes_block1.size // nbits1
npatches2 = indexes_block2.size // indexes_block2.shape[-1] # kernel size




dataset = np.random.randint(0,2,(100,20,20))
inputs = [x for x in dataset]

def preprocess_inputs(x, in_shape=(20,20)):

    x = np.tile(np.reshape(x,(1,*in_shape)), (nfilters1,1,1))

    preprocessed_input = conv2D_unfold(x, ksize=(kw1,kh1), stride=stride1,pad=pad1)
    # for i, x_filt in enumerate(preprocessed_input):
    #     if i == 0:
    #         y = BitsToIntAFast(x_filt).flatten().astype(int)
    #     else:
    #         y = np.concatenate((y, BitsToIntAFast(x_filt).flatten().astype(int)))
    y = BitsToIntAFast(preprocessed_input).flatten().astype(int)
    return y

inputs_block1 = [preprocess_inputs(x) for x in inputs]

tables = []
lk = []
for i, lk_filter in enumerate(block1.transpose()):
    tables.extend([lk_filter] * npatches1)# on each filter
    lk.extend([fhe.LookupTable(lk_filter)] * npatches1)  # * npatches)
    # lk.extend([fhe.LookupTable(lk_filter*np.random.randint(0,16,1))] * npatches)  # npatches per filter
tables_block1 = fhe.LookupTable(lk)


# print(block1)
lk = []
im = []
kernel = [2**(i) for i in range(nbits2)]


for i, t in enumerate(tables):
    row, col = i // height_input_block2, i % height_input_block2
    idx_kernel = (row%kw2)*2 + col%kh2
    lk.append(fhe.LookupTable(t * kernel[idx_kernel]))
    im.append(max(lk[-1]))
# print(np.array(im).reshape((width_input_block2, height_input_block2)))
tables_block1_temp = fhe.LookupTable(lk)

lk = []
all_tables = []
for i, lk_filter in enumerate(block2.transpose()):  # on each filter
    lk.extend([fhe.LookupTable(lk_filter)] * npatches2)  # * npatches)
    positions_filters = [t for t in
                         np.tile(lk_filter, s1*s2 // nfilters2).reshape((s1*s2 // nfilters2, 2 ** nbits2))]
    all_tables.extend(positions_filters)
    # lk.extend([fhe.LookupTable(lk_filter*np.random.randint(0,16,1))] * npatches)  # npatches per filter
tables_block2 = fhe.LookupTable(lk)

all_tables = np.array((all_tables))

w_tables = [(w_quant[i,:].copy().reshape(-1,1) * all_tables) for i in range(nclasses)]
tables_clear = [([(table_filter) for table_filter in w_tables[i].copy()]) for i in range(nclasses)]
tables = [fhe.LookupTable([fhe.LookupTable(table_filter) for table_filter in w_tables[i].copy()]) for i in range(nclasses)]
# lu_layer2 = fhe.LookupTable(tables)

tables_clear = np.array(tables_clear).reshape((nclasses*s1*s2*nfilters2, 2**nbits2))



split_lr = False



@fhe.compiler({"x": "encrypted"})
def g(x): # normal
    x = tables_block1[x]
    x = x.reshape(nfilters1, width_input_block2, height_input_block2)
    x = conv2D_unfold_deep(x, 1, ksize=(kw2, kh2), pad=pad2, stride=stride2)


    y = BitsToIntAFast(x).flatten().astype(int)

    y = tables_block2[y]

    # print(res_f.shape)
    return y @ w_quant.transpose()  # np.greater(res,0)#res #(res>0)*1.0


@fhe.compiler({"x": "encrypted"})
def j(x): # output of tables as indexes
    x = tables_block1_temp[x]
    x = x.reshape(nfilters1, width_input_block2, height_input_block2)
    x = conv2D_unfold_deep(x, 1, ksize=(kw2, kh2), pad=pad2, stride=stride2)


    y = np.sum(x, axis=-1).flatten()

    y = tables_block2[y]

    # print(res_f.shape)
    return y @ w_quant.transpose()  # np.greater(res,0)#res #(res>0)*1.0



# 64*11*11*4
# key_path = os.path.join('./keys', 'mnist_full_pr')
# if not os.path.exists(key_path):
#     os.mkdir(key_path)

inputset = inputs_block1[:10]
inpt = [inputs_block1[11]]

# g(*inpt)
# j(*inpt)
# k(*inpt)
# l(*inpt)



cfg = fhe.Configuration(enable_unsafe_features=True,
                                  show_mlir=False,
                                  show_progress=True,
                                  progress_tag=True,
                                  progress_title='Evaluation')

multi = True

dataset_name = 'debug_discord' # 'MNIST'
model_name = f'lr_{nbitslr}_2Blocks'
mode_preprocess = 'fullpr'
split_lr_str = '_split_lr' if split_lr else ''
multi_precision = '_multiprecision' if multi else ''

log_path = dataset_name + '_' + model_name + '_' + mode_preprocess + split_lr_str + multi_precision + '_normal'
log_dir = os.path.join('./logs', dataset_name)
key_dir = os.path.join('./keys', dataset_name, log_path)
logger = create_logger(log_path, log_dir=log_dir)

logger.info(f"Logs will be saved at:\t{os.path.join(log_dir, log_path)}\n")

circuit, res, t_final = compile_keygen_run(g, inputset, inpt, log_path, logger, cfg, multi_precision=multi, log_dir=log_dir, key_dir=key_dir)


