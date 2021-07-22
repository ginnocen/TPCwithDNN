# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=fixme
import random
import numpy as np
import scipy.constants

from tpcwithdnn.logger import get_logger

SCALES_CONST = [0, 3, -3, 6, -6]
SCALES_LINEAR = [0, 3, -3]
SCALES_PARABOLIC = [0, 3, -3]
NUM_FOURIER_COEFFS = 40
NELE_PER_ADC = 670

def get_mean_desc(mean_id):
    s_const = SCALES_CONST[mean_id // 9]
    s_lin = SCALES_LINEAR[(mean_id % 9) // 3]
    s_para = SCALES_PARABOLIC[mean_id % 3]
    return "%d-Const_%d_Lin_%d_Para_%d" % (mean_id, s_const, s_lin, s_para)


def load_data_original_idc(dirinput, event_index, z_range):
    """
    Load IDC data.
    """
    mean_prefix = get_mean_desc(event_index[1])

    files = ["%s/Pos/vecRPos.npy" % dirinput,
             "%s/Pos/vecPhiPos.npy" % dirinput,
             "%s/Pos/vecZPos.npy" % dirinput,
             "%s/Mean/%s-vecMeanSC.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomSC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistR.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomDistR.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistRPhi.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomDistRPhi.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistZ.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomDistZ.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanCorrR.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomCorrR.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanCorrRPhi.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomCorrRPhi.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanCorrZ.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomCorrZ.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-numMeanZeroDIDCA.npy" % (dirinput, mean_prefix),
             "%s/Mean/%s-numMeanZeroDIDCC.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-numRandomZeroDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-numRandomZeroDIDCC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanOneDIDCA.npy" % (dirinput, mean_prefix),
             "%s/Mean/%s-vecMeanOneDIDCC.npy" % (dirinput, mean_prefix),
             "%s/Random/%d-vecRandomOneDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-vecRandomOneDIDCC.npy" % (dirinput, event_index[0])]

    vec_z_pos_tmp = np.load(files[2])
    vec_sel_in_z = (z_range[0] <= vec_z_pos_tmp) & (vec_z_pos_tmp < z_range[1])

    mean_plus_prefix = "9-Const_3_Lin_0_Para_0"
    mean_minus_prefix = "18-Const_-3_Lin_0_Para_0"

    ref_mean_sc_plus_file = "%s/Mean/%s-vecMeanSC.npy" % (dirinput, mean_plus_prefix)
    ref_mean_sc_minus_file = "%s/Mean/%s-vecMeanSC.npy" % (dirinput, mean_minus_prefix)
    vec_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_in_z] - \
        np.load(ref_mean_sc_minus_file)[vec_sel_in_z]

    mat_der_ref_mean_corr = np.empty((3, vec_der_ref_mean_sc.size))
    ref_mean_corr_r_plus_file = "%s/Mean/%s-vecMeanCorrR.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_r_minus_file = "%s/Mean/%s-vecMeanCorrR.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[0, :] = np.load(ref_mean_corr_r_plus_file)[vec_sel_in_z] \
        - np.load(ref_mean_corr_r_minus_file)[vec_sel_in_z]
    ref_mean_corr_rphi_plus_file = "%s/Mean/%s-vecMeanCorrRPhi.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_rphi_minus_file = "%s/Mean/%s-vecMeanCorrRPhi.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[1, :] = np.load(ref_mean_corr_rphi_plus_file)[vec_sel_in_z] - \
        np.load(ref_mean_corr_rphi_minus_file)[vec_sel_in_z]
    ref_mean_corr_z_plus_file = "%s/Mean/%s-vecMeanCorrZ.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_z_minus_file = "%s/Mean/%s-vecMeanCorrZ.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[2, :] = np.load(ref_mean_corr_z_plus_file)[vec_sel_in_z] \
        - np.load(ref_mean_corr_z_minus_file)[vec_sel_in_z]

    data = [np.load(f)[vec_sel_in_z] for f in files[:-8]] + \
        [vec_der_ref_mean_sc, mat_der_ref_mean_corr] + \
        [np.load(f) for f in files[-8:]]
    return data

def filter_idc_data(data_a, data_c, z_range):
    # TODO: Getter and application of Fourier coefficients need to be modified to handle both A and
    # C side at the same time
    if z_range[0] < 0 and z_range[1] > 0:  # pylint: disable=chained-comparison
        logger = get_logger()
        logger.fatal("Framework not yet fully prepared to use data from both A and C side at once.")

    output_data = []
    for data in data_a:
        output_data.append([])
    if z_range[1] > 0:
        for ind, data in enumerate(data_a):
            output_data[ind] = np.hstack((output_data[ind],
                                          data / (scipy.constants.e * NELE_PER_ADC))) # C -> ADC
    if z_range[0] < 0:
        for ind, data in enumerate(data_c):
            output_data[ind] = np.hstack((output_data[ind],
                                          data / (scipy.constants.e * NELE_PER_ADC))) # C -> ADC
    return tuple(output_data)

def load_data_original(input_data, event_index):
    """
    Load old SC data.
    """
    files = ["%s/data/Pos/0-vecRPos.npy" % input_data,
             "%s/data/Pos/0-vecPhiPos.npy" % input_data,
             "%s/data/Pos/0-vecZPos.npy" % input_data,
             "%s/data/Mean/%d-vecMeanSC.npy" % (input_data, event_index[1]),
             "%s/data/Random/%d-vecRandomSC.npy" % (input_data, event_index[0]),
             "%s/data/Mean/%d-vecMeanDistR.npy" % (input_data, event_index[1]),
             "%s/data/Random/%d-vecRandomDistR.npy" % (input_data, event_index[0]),
             "%s/data/Mean/%d-vecMeanDistRPhi.npy" % (input_data, event_index[1]),
             "%s/data/Random/%d-vecRandomDistRPhi.npy" % (input_data, event_index[0]),
             "%s/data/Mean/%d-vecMeanDistZ.npy" % (input_data, event_index[1]),
             "%s/data/Random/%d-vecRandomDistZ.npy" % (input_data, event_index[0])]

    return [np.load(f) for f in files]

def load_data_derivatives_ref_mean(inputdata, z_range):
    z_pos_file = "%s/data/Pos/0-vecZPos.npy" % inputdata
    ref_mean_sc_plus_file = "%s/data/Mean/9-vecMeanSC.npy" % inputdata
    ref_mean_sc_minus_file = "%s/data/Mean/18-vecMeanSC.npy" % inputdata

    vec_z_pos = np.load(z_pos_file)
    vec_sel_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    vec_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_z] - \
                          np.load(ref_mean_sc_minus_file)[vec_sel_z]

    mat_der_ref_mean_dist = np.empty((3, vec_der_ref_mean_sc.size))
    ref_mean_dist_r_plus_file = "%s/data/Mean/9-vecMeanDistR.npy" % inputdata
    ref_mean_dist_r_minus_file = "%s/data/Mean/18-vecMeanDistR.npy" % inputdata
    mat_der_ref_mean_dist[0, :] = np.load(ref_mean_dist_r_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_dist_r_minus_file)[vec_sel_z]
    ref_mean_dist_rphi_plus_file = "%s/data/Mean/9-vecMeanDistRPhi.npy" % inputdata
    ref_mean_dist_rphi_minus_file = "%s/data/Mean/18-vecMeanDistRPhi.npy" % inputdata
    mat_der_ref_mean_dist[1, :] = np.load(ref_mean_dist_rphi_plus_file)[vec_sel_z] - \
                                                np.load(ref_mean_dist_rphi_minus_file)[vec_sel_z]
    ref_mean_dist_z_plus_file = "%s/data/Mean/9-vecMeanDistZ.npy" % inputdata
    ref_mean_dist_z_minus_file = "%s/data/Mean/18-vecMeanDistZ.npy" % inputdata
    mat_der_ref_mean_dist[2, :] = np.load(ref_mean_dist_z_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_dist_z_minus_file)[vec_sel_z]

    return vec_der_ref_mean_sc, mat_der_ref_mean_dist

def mat_to_vec(opt_pred, mat_tuple):
    if sum(opt_pred) > 1:
        logger = get_logger()
        logger.fatal("Framework not yet fully prepared for more than one distortion direction.")

    sel_opts = np.array(opt_pred) > 0
    res = tuple(np.hstack(mat[sel_opts]) for mat in mat_tuple)
    return res

def downsample_data(data_size, downsample_frac):
    chosen = [False] * data_size
    num_points = int(round(downsample_frac * data_size))
    for _ in range(num_points):
        sel_ind = random.randrange(0, data_size)
        while chosen[sel_ind]:
            sel_ind = random.randrange(0, data_size)
        chosen[sel_ind] = True
    return chosen

def get_fourier_coeffs(vec_oned_idc):
    dft = np.fft.fft(vec_oned_idc)
    dft_real = np.real(dft)[:NUM_FOURIER_COEFFS]
    dft_imag = np.imag(dft)[:NUM_FOURIER_COEFFS]

    return np.concatenate((dft_real, dft_imag))


def get_input_oned_idc_single_map(vec_r_pos, vec_phi_pos, vec_z_pos,
                                  vec_der_ref_mean_corr, dft_coeffs):
    inputs = np.zeros((vec_der_ref_mean_corr.size,
                       4 + dft_coeffs.size))
    for ind, pos in enumerate((vec_r_pos, vec_phi_pos, vec_z_pos)):
        inputs[:, ind] = pos
    inputs[:, 3] = vec_der_ref_mean_corr
    inputs[:, -dft_coeffs.size:] = dft_coeffs  # pylint: disable=invalid-unary-operand-type
    return inputs


def get_input_names_oned_idc():
    # input_names = ['r', 'phi', 'z', 'der_corr_r', 'fluc_0d_idc']
    input_names = ['r', 'phi', 'z', 'der_corr_r']
    input_names = input_names + ['c_real%d' % i for i in range(0, NUM_FOURIER_COEFFS)] + \
        ['c_imag%d' % i for i in range(0, NUM_FOURIER_COEFFS)]
    return input_names


def load_data_oned_idc(dirinput, event_index, z_range,
                      opt_pred, downsample, downsample_frac):
    [vec_r_pos, vec_phi_pos, vec_z_pos,
     *_,
     vec_mean_corr_r, vec_random_corr_r,
     vec_mean_corr_phi, vec_random_corr_phi,
     vec_mean_corr_z, vec_random_corr_z,
     _, mat_der_ref_mean_corr,
     _, _, _, _,
     vec_mean_oned_idc_a, vec_mean_oned_idc_c,
     vec_random_oned_idc_a, vec_random_oned_idc_c] = load_data_original_idc(dirinput, event_index,
                                                                  z_range)

    vec_oned_idc_fluc,  = filter_idc_data( # pylint: disable=unbalanced-tuple-unpacking
              (vec_random_oned_idc_a - vec_mean_oned_idc_a, ),
              (vec_random_oned_idc_c - vec_mean_oned_idc_c, ), z_range)
    dft_coeffs = get_fourier_coeffs(vec_oned_idc_fluc)

    mat_fluc_corr = np.array((vec_random_corr_r - vec_mean_corr_r,
                              vec_random_corr_phi - vec_mean_corr_phi,
                              vec_random_corr_z - vec_mean_corr_z))

    vec_exp_corr_fluc, vec_der_ref_mean_corr =\
        mat_to_vec(opt_pred, (mat_fluc_corr, mat_der_ref_mean_corr))
    # TODO: this will not work properly if vec_exp_corr_fluc containes more than one
    # distortion direction
    if downsample:
        chosen_points = downsample_data(len(vec_z_pos), downsample_frac)
        vec_r_pos = vec_r_pos[chosen_points]
        vec_phi_pos = vec_phi_pos[chosen_points]
        vec_z_pos = vec_z_pos[chosen_points]
        vec_der_ref_mean_corr = vec_der_ref_mean_corr[chosen_points]
        vec_exp_corr_fluc = vec_exp_corr_fluc[chosen_points]

    inputs = get_input_oned_idc_single_map(vec_r_pos, vec_phi_pos, vec_z_pos,
                                           vec_der_ref_mean_corr, dft_coeffs)

    return inputs, vec_exp_corr_fluc


def load_data(input_data, event_index, z_range):

    """ Here we define the functionalties to load the files from the input
    directory which is set in the database. Here below the description of
    the input files:
        - 0-vecZPos.npy, 0-vecRPos.npy, 0-vecPhiPos.npy contains the
        position of the FIXME. There is only one of these files for each
        folder, therefore for each bunch of events
        Input features for training:
        - vecMeanSC.npy: average space charge in each bin of r, rphi and z.
        - vecRandomSC.npy: fluctuation of the space charge.
        Output from the numberical calculations:
        - vecMeanDistR.npy average distorsion along the R axis in the same
          grid. It represents the expected distorsion that an electron
          passing by that region would have as a consequence of the IBF.
        - vecRandomDistR.npy are the correponding fluctuations.
        - All the distorsions along the other directions have a consistent
          naming choice.

    """

    [_, _, vec_z_pos,
     vec_mean_sc, vec_random_sc,
     vec_mean_dist_r, vec_random_dist_r,
     vec_mean_dist_rphi, vec_random_dist_rphi,
     vec_mean_dist_z, vec_random_dist_z] = load_data_original(input_data, event_index)

    vec_sel_in_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    vec_mean_sc = vec_mean_sc[vec_sel_in_z]
    vec_fluctuation_sc = vec_random_sc[vec_sel_in_z] - vec_mean_sc

    vec_fluctuation_dist_r = vec_random_dist_r[vec_sel_in_z] - vec_mean_dist_r[vec_sel_in_z]
    vec_fluctuation_dist_rphi = vec_random_dist_rphi[vec_sel_in_z] -\
        vec_mean_dist_rphi[vec_sel_in_z]
    vec_fluctuation_dist_z = vec_random_dist_z[vec_sel_in_z] - vec_mean_dist_z[vec_sel_in_z]

    return [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
            vec_fluctuation_dist_rphi, vec_fluctuation_dist_z]


def load_event_idc(dirinput, event_index, z_range,
                   opt_pred, downsample, downsample_frac):

    inputs, exp_outputs = load_data_oned_idc(dirinput, event_index, z_range,
                                            opt_pred, downsample, downsample_frac)

    dim_output = sum(opt_pred)
    if dim_output > 1:
        logger = get_logger()
        logger.fatal("YOU CAN PREDICT ONLY 1 DISTORTION. The sum of opt_predout == 1")

    #print("DIMENSION INPUT TRAINING", inputs.shape)
    #print("DIMENSION OUTPUT TRAINING", exp_outputs.shape)

    return inputs, exp_outputs


def load_train_apply(input_data, event_index, z_range,
                     grid_r, grid_rphi, grid_z, opt_train, opt_pred):

    [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
     vec_fluctuation_dist_rphi, vec_fluctuation_dist_z] = \
        load_data(input_data, event_index, z_range)
    dim_input = sum(opt_train)
    dim_output = sum(opt_pred)
    inputs = np.empty((grid_rphi, grid_r, grid_z, dim_input))
    exp_outputs = np.empty((grid_rphi, grid_r, grid_z, dim_output))

    indexfillx = 0
    if opt_train[0] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_mean_sc.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1
    if opt_train[1] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_fluctuation_sc.reshape(grid_rphi, grid_r, grid_z)

    if dim_output > 1:
        logger = get_logger()
        logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")

    flucs = np.array((vec_fluctuation_dist_r, vec_fluctuation_dist_rphi, vec_fluctuation_dist_z))
    sel_flucs = flucs[np.array(opt_pred) == 1]
    for ind, vec_fluctuation_dist in enumerate(sel_flucs):
        exp_outputs[:, :, :, ind] = \
                vec_fluctuation_dist.reshape(grid_rphi, grid_r, grid_z)

    #print("DIMENSION INPUT TRAINING", inputs.shape)
    #print("DIMENSION OUTPUT TRAINING", exp_outputs.shape)

    return inputs, exp_outputs


def get_event_mean_indices(maxrandomfiles, range_mean_index, ranges):
    all_indices_events_means = []
    for ievent in np.arange(maxrandomfiles):
        for imean in np.arange(range_mean_index[0], range_mean_index[1] + 1):
            all_indices_events_means.append([ievent, imean])
    # Equivalent to shuffling the data
    sel_indices_events_means = random.sample(all_indices_events_means, \
        maxrandomfiles * (range_mean_index[1] + 1 - range_mean_index[0]))

    indices_train = sel_indices_events_means[ranges["train"][0]:ranges["train"][1]]
    indices_val = sel_indices_events_means[ranges["val"][0]:ranges["val"][1]]
    indices_apply = sel_indices_events_means[ranges["apply"][0]:ranges["apply"][1]]

    partition = {"train": indices_train,
                 "validation": indices_val,
                 "apply": indices_apply}

    return sel_indices_events_means, partition
