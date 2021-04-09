# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=fixme
import random
import numpy as np

from tpcwithdnn.logger import get_logger

def load_data_original_idc(dirinput, event_index):
    """
    Load IDC data.
    """
    # TODO: How to load both A and C side data if both needed? Concatenate? Get average?
    mean_factors = [100, 97, 94, 91, 103, 106, 109]
    mean_factor = mean_factors[event_index[1]]
    files = ["%s/Pos/vecRPos.npy" % dirinput,
             "%s/Pos/vecPhiPos.npy" % dirinput,
             "%s/Pos/vecZPos.npy" % dirinput,
             "%s/Mean/%d-%d-numMeanZeroDIDCA.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Mean/%d-%d-numMeanZeroDIDCC.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-numRandomZeroDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-numRandomZeroDIDCC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanOneDIDCA.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Mean/%d-%d-vecMeanOneDIDCC.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomOneDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-vecRandomOneDIDCC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanSC.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomSC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanDistR.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomDistR.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanDistRPhi.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomDistRPhi.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanDistZ.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomDistZ.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanCorrR.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomCorrR.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanCorrRPhi.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomCorrRPhi.npy" % (dirinput, event_index[0]),
             "%s/Mean/%d-%d-vecMeanCorrZ.npy" % (dirinput, event_index[1], mean_factor),
             "%s/Random/%d-vecRandomCorrZ.npy" % (dirinput, event_index[0])]

    return [np.load(f) for f in files]

def filter_idc_data(data_a, data_c, z_range):
    output_data = []
    for data in data_a:
        output_data.append([])
    if z_range[1] > 0:
        for ind, data in enumerate(data_a):
            output_data[ind] = np.hstack((output_data[ind], data))
    if z_range[0] < 0:
        for ind, data in enumerate(data_c):
            output_data[ind] = np.hstack((output_data[ind], data))
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

def load_data_derivatives_ref_mean_idc(dirinput, z_range):
    z_pos_file = "%s/Pos/vecZPos.npy" % dirinput
    ref_mean_sc_plus_file = "%s/Mean/5-106-vecMeanSC.npy" % dirinput
    ref_mean_sc_minus_file = "%s/Mean/2-94-vecMeanSC.npy" % dirinput

    vec_z_pos = np.load(z_pos_file)
    vec_sel_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    arr_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_z] - \
                          np.load(ref_mean_sc_minus_file)[vec_sel_z]

    mat_der_ref_mean_corr = np.empty((3, arr_der_ref_mean_sc.size))
    ref_mean_corr_r_plus_file = "%s/Mean/5-106-vecMeanDistR.npy" % dirinput
    ref_mean_corr_r_minus_file = "%s/Mean/2-94-vecMeanDistR.npy" % dirinput
    mat_der_ref_mean_corr[0, :] = np.load(ref_mean_corr_r_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_corr_r_minus_file)[vec_sel_z]
    ref_mean_corr_rphi_plus_file = "%s/Mean/5-106-vecMeanDistRPhi.npy" % dirinput
    ref_mean_corr_rphi_minus_file = "%s/Mean/2-94-vecMeanDistRPhi.npy" % dirinput
    mat_der_ref_mean_corr[1, :] = np.load(ref_mean_corr_rphi_plus_file)[vec_sel_z] - \
                                                np.load(ref_mean_corr_rphi_minus_file)[vec_sel_z]
    ref_mean_corr_z_plus_file = "%s/Mean/5-106-vecMeanDistZ.npy" % dirinput
    ref_mean_corr_z_minus_file = "%s/Mean/2-94-vecMeanDistZ.npy" % dirinput
    mat_der_ref_mean_corr[2, :] = np.load(ref_mean_corr_z_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_corr_z_minus_file)[vec_sel_z]

    return arr_der_ref_mean_sc, mat_der_ref_mean_corr

def load_data_derivatives_ref_mean(inputdata, z_range):
    z_pos_file = "%s/data/Pos/0-vecZPos.npy" % inputdata
    ref_mean_sc_plus_file = "%s/data/Mean/9-vecMeanSC.npy" % inputdata
    ref_mean_sc_minus_file = "%s/data/Mean/18-vecMeanSC.npy" % inputdata

    vec_z_pos = np.load(z_pos_file)
    vec_sel_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    arr_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_z] - \
                          np.load(ref_mean_sc_minus_file)[vec_sel_z]

    mat_der_ref_mean_dist = np.empty((3, arr_der_ref_mean_sc.size))
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

    return arr_der_ref_mean_sc, mat_der_ref_mean_dist


def load_data_one_idc(dirinput, event_index, input_z_range, output_z_range, opt_pred):
    [_, _, vec_z_pos,
     num_mean_zero_idc_a, num_mean_zero_idc_c, num_random_zero_idc_a, num_random_zero_idc_c,
     vec_mean_one_idc_a, vec_mean_one_idc_c, vec_random_one_idc_a, vec_random_one_idc_c,
     *_,
     vec_mean_corr_r, vec_random_corr_r,
     vec_mean_corr_rphi, vec_random_corr_rphi,
     vec_mean_corr_z, vec_random_corr_z] = load_data_original_idc(dirinput, event_index)

    vec_sel_out_z = (output_z_range[0] <= vec_z_pos) & (vec_z_pos < output_z_range[1])

    data_a = (vec_random_one_idc_a - vec_mean_one_idc_a,
              num_random_zero_idc_a - num_mean_zero_idc_a)
    data_c = (vec_random_one_idc_c - vec_mean_one_idc_c,
              num_random_zero_idc_c - num_mean_zero_idc_c)
    vec_one_idc_fluc, num_zero_idc_fluc = filter_idc_data(data_a, data_c, input_z_range) # pylint: disable=unbalanced-tuple-unpacking

    mat_mean_corr = (vec_mean_corr_r, vec_mean_corr_rphi, vec_mean_corr_z)
    mat_random_corr = (vec_random_corr_r, vec_random_corr_rphi, vec_random_corr_z)
    _, mat_der_ref_mean_corr = load_data_derivatives_ref_mean_idc(dirinput, input_z_range)

    vec_mean_corr = []
    vec_random_corr = []
    vec_der_ref_mean_corr = []
    for ind, (vec_mean, vec_random, vec_der_ref_mean) in \
        enumerate(zip(mat_mean_corr, mat_random_corr, mat_der_ref_mean_corr)):
        if opt_pred[ind] == 1:
            vec_mean_corr = np.hstack((vec_mean_corr, vec_mean))
            vec_random_corr = np.hstack((vec_random_corr, vec_random))
            vec_der_ref_mean_corr = np.hstack((vec_der_ref_mean_corr, vec_der_ref_mean))

    vec_exp_corr_fluc = vec_random_corr - vec_mean_corr
    vec_exp_corr_fluc = vec_exp_corr_fluc[vec_sel_out_z]

    return vec_one_idc_fluc, vec_der_ref_mean_corr, num_zero_idc_fluc, vec_exp_corr_fluc


def load_data(input_data, event_index, input_z_range, output_z_range):

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

    vec_sel_in_z = (input_z_range[0] <= vec_z_pos) & (vec_z_pos < input_z_range[1])
    vec_sel_out_z = (output_z_range[0] <= vec_z_pos) & (vec_z_pos < output_z_range[1])

    vec_mean_sc = vec_mean_sc[vec_sel_in_z]
    vec_fluctuation_sc = vec_random_sc[vec_sel_in_z] - vec_mean_sc

    vec_fluctuation_dist_r = vec_random_dist_r[vec_sel_out_z] - vec_mean_dist_r[vec_sel_out_z]
    vec_fluctuation_dist_rphi = vec_random_dist_rphi[vec_sel_out_z] -\
                                     vec_mean_dist_rphi[vec_sel_out_z]
    vec_fluctuation_dist_z = vec_random_dist_z[vec_sel_out_z] - vec_mean_dist_z[vec_sel_out_z]

    return [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
            vec_fluctuation_dist_rphi, vec_fluctuation_dist_z]


def load_train_apply_idc(dirinput, event_index, input_z_range, output_z_range,
                         opt_pred):

    (vec_one_idc_fluc, vec_der_ref_mean_corr, num_zero_idc_fluc, exp_outputs) =\
        load_data_one_idc(dirinput, event_index, input_z_range, output_z_range, opt_pred)

    inputs = np.array([[*vec_one_idc_fluc, num_der, *num_zero_idc_fluc]
                        for num_der in vec_der_ref_mean_corr])

    dim_output = sum(opt_pred)
    if dim_output > 1:
        logger = get_logger()
        logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")

    #print("DIMENSION INPUT TRAINING", inputs.shape)
    #print("DIMENSION OUTPUT TRAINING", exp_outputs.shape)

    return inputs, exp_outputs


def load_train_apply(input_data, event_index, input_z_range, output_z_range,
                     grid_r, grid_rphi, grid_z, opt_train, opt_pred):

    [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
     vec_fluctuation_dist_rphi, vec_fluctuation_dist_z] = \
        load_data(input_data, event_index, input_z_range, output_z_range)
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
    sel_indices_events_means = random.sample(all_indices_events_means, \
        maxrandomfiles * (range_mean_index[1] + 1 - range_mean_index[0]))

    indices_train = sel_indices_events_means[ranges["train"][0]:ranges["train"][1]]
    indices_test = sel_indices_events_means[ranges["test"][0]:ranges["test"][1]]
    indices_apply = sel_indices_events_means[ranges["apply"][0]:ranges["apply"][1]]

    partition = {"train": indices_train,
                 "validation": indices_test,
                 "apply": indices_apply}

    return sel_indices_events_means, partition
