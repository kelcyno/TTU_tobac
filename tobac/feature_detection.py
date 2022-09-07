import logging
import numpy as np
import pandas as pd
from . import utils as tb_utils
import warnings


def feature_position(
    hdim1_indices,
    hdim2_indices,
    region_small=None,
    region_bbox=None,
    track_data=None,
    threshold_i=None,
    position_threshold="center",
    target=None,
):
    """Function to  determine feature position

    Parameters
    ----------
    hdim1_indices : list
        list of indices along hdim1 (typically ```y```)

    hdim2_indices : list
        List of indices of feature along hdim2 (typically ```x```)

    region_small : 2D array-like
        A true/false array containing True where the threshold
        is met and false where the threshold isn't met. This array should
        be the the size specified by region_bbox, and can be a subset of the
        overall input array (i.e., ```track_data```).

    region_bbox : list or tuple with length of 4
        The coordinates that region_small occupies within the total track_data
        array. This is in the order that the coordinates come from the
        ```get_label_props_in_dict``` function. For 2D data, this should be:
        (hdim1 start, hdim 2 start, hdim 1 end, hdim 2 end).

    track_data : 2D array-like
        2D array containing the data

    threshold_i : float
        The threshold value that we are testing against

    position_threshold : {'center', 'extreme', 'weighted_diff', 'weighted abs'}
        How to select the single point position from our data.
        'center' picks the geometrical centre of the region, and is typically not recommended.
        'extreme' picks the maximum or minimum value inside the region (max/min set by ```target```)
        'weighted_diff' picks the centre of the region weighted by the distance from the threshold value
        'weighted_abs' picks the centre of the region weighted by the absolute values of the field

    target : {'maximum', 'minimum'}
        Used only when position_threshold is set to 'extreme', this sets whether
        it is looking for maxima or minima.

    Returns
    -------
    float
            feature position along 1st horizontal dimension
    float
            feature position along 2nd horizontal dimension
    """

    track_data_region = track_data[
        region_bbox[0] : region_bbox[2], region_bbox[1] : region_bbox[3]
    ]

    if position_threshold == "center":
        # get position as geometrical centre of identified region:
        hdim1_index = np.mean(hdim1_indices)
        hdim2_index = np.mean(hdim2_indices)

    elif position_threshold == "extreme":
        # get position as max/min position inside the identified region:
        if target == "maximum":
            index = np.argmax(track_data_region[region_small])
        if target == "minimum":
            index = np.argmin(track_data_region[region_small])
        hdim1_index = hdim1_indices[index]
        hdim2_index = hdim2_indices[index]

    elif position_threshold == "weighted_diff":
        # get position as centre of identified region, weighted by difference from the threshold:
        weights = abs(track_data_region[region_small] - threshold_i)
        if sum(weights) == 0:
            weights = None
        hdim1_index = np.average(hdim1_indices, weights=weights)
        hdim2_index = np.average(hdim2_indices, weights=weights)

    elif position_threshold == "weighted_abs":
        # get position as centre of identified region, weighted by absolute values if the field:
        weights = abs(track_data_region[region_small])
        if sum(weights) == 0:
            weights = None
        hdim1_index = np.average(hdim1_indices, weights=weights)
        hdim2_index = np.average(hdim2_indices, weights=weights)

    else:
        raise ValueError(
            "position_threshold must be center,extreme,weighted_diff or weighted_abs"
        )
    return hdim1_index, hdim2_index


def test_overlap(region_inner, region_outer):
    """
    function to test for overlap between two regions (probably scope for further speedup here)
    Input:
    region_1:      list
                   list of 2-element tuples defining the indices of all cell in the region
    region_2:      list
                   list of 2-element tuples defining the indices of all cell in the region

    Output:
    overlap:       bool
                   True if there are any shared points between the two regions
    """
    overlap = frozenset(region_outer).isdisjoint(region_inner)
    return not overlap


def remove_parents(features_thresholds, regions_i, regions_old):
    """
    function to remove features whose regions surround newly detected feature regions
    Input:
        features_thresholds:    pandas.DataFrame
                                Dataframe containing detected features
    regions_i:                  dict
                                dictionary containing the regions above/below threshold for the newly detected feature (feature ids as keys)
    regions_old:                dict
                                dictionary containing the regions above/below threshold from previous threshold (feature ids as keys)
    Output:
        features_thresholds     pandas.DataFrame
                                Dataframe containing detected features excluding those that are superseded by newly detected ones
    """
    try:
        all_curr_pts = np.concatenate([vals for idx, vals in regions_i.items()])
        all_old_pts = np.concatenate([vals for idx, vals in regions_old.items()])
    except ValueError:
        # the case where there are no regions
        return features_thresholds
    old_feat_arr = np.empty((len(all_old_pts)))
    curr_loc = 0
    for idx_old in regions_old:
        old_feat_arr[curr_loc : curr_loc + len(regions_old[idx_old])] = idx_old
        curr_loc += len(regions_old[idx_old])

    common_pts, common_ix_new, common_ix_old = np.intersect1d(
        all_curr_pts, all_old_pts, return_indices=True
    )
    list_remove = np.unique(old_feat_arr[common_ix_old])

    # remove parent regions:
    if features_thresholds is not None:
        features_thresholds = features_thresholds[
            ~features_thresholds["idx"].isin(list_remove)
        ]

    return features_thresholds


def feature_detection_threshold(
    data_i,
    i_time,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    idx_start=0,
):
    """
    function to find features based on individual threshold value:
    Input:
    data_i:      iris.cube.Cube
                 2D field to perform the feature detection (single timestep)
    i_time:      int
                 number of the current timestep
    threshold:    float
                  threshold value used to select target regions to track
    target:       str ('minimum' or 'maximum')
                  flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    idx_start: int
               feature id to start with
    Output:
    features_threshold:      pandas DataFrame
                             detected features for individual threshold
    regions:                 dict
                             dictionary containing the regions above/below threshold used for each feature (feature ids as keys)
    """
    from skimage.measure import label
    from skimage.morphology import binary_erosion
    from copy import deepcopy

    if min_num != 0:
        warnings.warn(
            "min_num parameter has no effect and will be deprecated in a future version of tobac. Please use n_min_threshold instead",
            FutureWarning,
        )

    # if looking for minima, set values above threshold to 0 and scale by data minimum:
    if target == "maximum":
        mask = 1 * (data_i >= threshold)
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
    elif target == "minimum":
        mask = 1 * (data_i <= threshold)
    # only include values greater than threshold
    # erode selected regions by n pixels
    if n_erosion_threshold > 0:
        selem = np.ones((n_erosion_threshold, n_erosion_threshold))
        mask = binary_erosion(mask, selem).astype(np.int64)
        # detect individual regions, label  and count the number of pixels included:
    labels, num_labels = label(mask, background=0, return_num=True)

    label_props = tb_utils.get_label_props_in_dict(labels)
    if len(label_props) > 0:
        (
            total_indices_all,
            hdim1_indices_all,
            hdim2_indices_all,
        ) = tb_utils.get_indices_of_labels_from_reg_prop_dict(label_props)

    x_size = labels.shape[1]

    # check if not entire domain filled as one feature
    if num_labels > 0:
        # create empty list to store individual features for this threshold
        list_features_threshold = list()
        # create empty dict to store regions for individual features for this threshold
        regions = dict()

        # loop over individual regions:
        for cur_idx in total_indices_all:
            # skip this if there aren't enough points to be considered a real feature
            # as defined above by n_min_threshold
            curr_count = total_indices_all[cur_idx]
            if curr_count <= n_min_threshold:
                continue

            hdim1_indices = hdim1_indices_all[cur_idx]
            hdim2_indices = hdim2_indices_all[cur_idx]

            # Get location and size of the minimum bounding box
            # that will cover the whole labeled region
            label_bbox = label_props[cur_idx].bbox
            bbox_ystart, bbox_xstart, bbox_yend, bbox_xend = label_bbox
            bbox_xsize = bbox_xend - bbox_xstart
            bbox_ysize = bbox_yend - bbox_ystart

            # Create a smaller region that is the maxium extent in X and Y
            # of the labeled point so that we don't have to do operations
            # on the full array
            region_small = np.full((bbox_ysize, bbox_xsize), False)
            region_small[
                hdim1_indices - bbox_ystart, hdim2_indices - bbox_xstart
            ] = True

            # Later on, rather than doing (more expensive) operations
            # on tuples, let's convert to 1D coordinates.
            region_i = np.array(hdim1_indices * x_size + hdim2_indices)

            regions[cur_idx + idx_start] = region_i

            single_indices = feature_position(
                hdim1_indices,
                hdim2_indices,
                region_small=region_small,
                region_bbox=label_bbox,
                track_data=data_i,
                threshold_i=threshold,
                position_threshold=position_threshold,
                target=target,
            )

            hdim1_index, hdim2_index = single_indices
            # create individual DataFrame row in tracky format for identified feature
            list_features_threshold.append(
                {
                    "frame": int(i_time),
                    "idx": cur_idx + idx_start,
                    "hdim_1": hdim1_index,
                    "hdim_2": hdim2_index,
                    "num": curr_count,
                    "threshold_value": threshold,
                }
            )
            column_names = [
                "frame",
                "idx",
                "hdim_1",
                "hdim_2",
                "num",
                "threshold_value",
            ]

        # after looping thru proto-features, check if any exceed num threshold
        # if they do not, provide a blank pandas df and regions dict
        if list_features_threshold == []:
            features_threshold = pd.DataFrame()
            regions = dict()
        # if they do, provide a dataframe with features organized with 2D and 3D metadata
        else:
            features_threshold = pd.DataFrame(
                list_features_threshold, columns=column_names
            )
    else:
        features_threshold = pd.DataFrame()
        regions = dict()

    return features_threshold, regions


def feature_detection_multithreshold_timestep(
    data_i,
    i_time,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    feature_number_start=1,
):
    """
    function to find features in each timestep based on iteratively finding regions above/below a set of thresholds
    Input:
    data_i:      iris.cube.Cube
                 2D field to perform the feature detection (single timestep)
    i_time:      int
                 number of the current timestep

    threshold:    list of floats
                  threshold values used to select target regions to track
    dxy:          float
                  grid spacing of the input data (m)
    target:       str ('minimum' or 'maximum')
                  flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    feature_number_start: int
                          feature number to start with
    Output:
    features_threshold:      pandas DataFrame
                             detected features for individual timestep
    """
    # Handle scipy depreciation gracefully
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        from scipy.ndimage.filters import gaussian_filter

    if min_num != 0:
        warnings.warn(
            "min_num parameter has no effect and will be deprecated in a future version of tobac. Please use n_min_threshold instead",
            FutureWarning,
        )

    track_data = data_i.core_data()

    track_data = gaussian_filter(
        track_data, sigma=sigma_threshold
    )  # smooth data slightly to create rounded, continuous field
    # create empty lists to store regions and features for individual timestep
    features_thresholds = pd.DataFrame()
    for i_threshold, threshold_i in enumerate(threshold):
        if i_threshold > 0 and not features_thresholds.empty:
            idx_start = features_thresholds["idx"].max() + feature_number_start
        else:
            idx_start = 0
        features_threshold_i, regions_i = feature_detection_threshold(
            track_data,
            i_time,
            threshold=threshold_i,
            sigma_threshold=sigma_threshold,
            min_num=min_num,
            target=target,
            position_threshold=position_threshold,
            n_erosion_threshold=n_erosion_threshold,
            n_min_threshold=n_min_threshold,
            min_distance=min_distance,
            idx_start=idx_start,
        )
        if any([x is not None for x in features_threshold_i]):
            features_thresholds = pd.concat([features_thresholds, features_threshold_i])

        # For multiple threshold, and features found both in the current and previous step, remove "parent" features from Dataframe
        if i_threshold > 0 and not features_thresholds.empty and regions_old:
            # for each threshold value: check if newly found features are surrounded by feature based on less restrictive threshold
            features_thresholds = remove_parents(
                features_thresholds, regions_i, regions_old
            )
        regions_old = regions_i

        logging.debug(
            "Finished feature detection for threshold "
            + str(i_threshold)
            + " : "
            + str(threshold_i)
        )
    return features_thresholds


def feature_detection_multithreshold(
    field_in,
    dxy,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    feature_number_start=1,
):
    """Function to perform feature detection based on contiguous regions above/below a threshold
    Input:
    field_in:      iris.cube.Cube
                   2D field to perform the tracking on (needs to have coordinate 'time' along one of its dimensions)

    thresholds:    list of floats
                   threshold values used to select target regions to track
    dxy:           float
                   grid spacing of the input data (m)
    target:        str ('minimum' or 'maximum')
                   flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    Output:
    features:      pandas DataFrame
                   detected features
    """
    from .utils import add_coordinates

    if min_num != 0:
        warnings.warn(
            "min_num parameter has no effect and will be deprecated in a future version of tobac. Please use n_min_threshold instead",
            FutureWarning,
        )

    logging.debug("start feature detection based on thresholds")

    # create empty list to store features for all timesteps
    list_features_timesteps = []

    # loop over timesteps for feature identification:
    data_time = field_in.slices_over("time")

    # if single threshold is put in as a single value, turn it into a list
    if type(threshold) in [int, float]:
        threshold = [threshold]

    for i_time, data_i in enumerate(data_time):
        time_i = data_i.coord("time").units.num2date(data_i.coord("time").points[0])
        features_thresholds = feature_detection_multithreshold_timestep(
            data_i,
            i_time,
            threshold=threshold,
            sigma_threshold=sigma_threshold,
            min_num=min_num,
            target=target,
            position_threshold=position_threshold,
            n_erosion_threshold=n_erosion_threshold,
            n_min_threshold=n_min_threshold,
            min_distance=min_distance,
            feature_number_start=feature_number_start,
        )
        # check if list of features is not empty, then merge features from different threshold values
        # into one DataFrame and append to list for individual timesteps:
        if not features_thresholds.empty:
            # Loop over DataFrame to remove features that are closer than distance_min to each other:
            if min_distance > 0:
                features_thresholds = filter_min_distance(
                    features_thresholds, dxy, min_distance
                )
        list_features_timesteps.append(features_thresholds)

        logging.debug(
            "Finished feature detection for " + time_i.strftime("%Y-%m-%d_%H:%M:%S")
        )

    logging.debug("feature detection: merging DataFrames")
    # Check if features are detected and then concatenate features from different timesteps into one pandas DataFrame
    # If no features are detected raise error
    if any([not x.empty for x in list_features_timesteps]):
        features = pd.concat(list_features_timesteps, ignore_index=True)
        features["feature"] = features.index + feature_number_start
        #    features_filtered = features.drop(features[features['num'] < min_num].index)
        #    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
        features = add_coordinates(features, field_in)
    else:
        features = None
        logging.info("No features detected")
    logging.debug("feature detection completed")
    return features


def filter_min_distance(features, dxy, min_distance):
    """Function to perform feature detection based on contiguous regions above/below a threshold
    Input:
    features:      pandas DataFrame
                   features
    dxy:           float
                   horzontal grid spacing (m)
    min_distance:  float
                   minimum distance between detected features (m)
    Output:
    features:      pandas DataFrame
                   features
    """
    from itertools import combinations

    remove_list_distance = []
    # create list of tuples with all combinations of features at the timestep:
    indices = combinations(features.index.values, 2)
    # Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
    for index_1, index_2 in indices:
        if index_1 is not index_2:
            features.loc[index_1, "hdim_1"]
            distance = dxy * np.sqrt(
                (features.loc[index_1, "hdim_1"] - features.loc[index_2, "hdim_1"]) ** 2
                + (features.loc[index_1, "hdim_2"] - features.loc[index_2, "hdim_2"])
                ** 2
            )
            if distance <= min_distance:
                #                        logging.debug('distance<= min_distance: ' + str(distance))
                if (
                    features.loc[index_1, "threshold_value"]
                    > features.loc[index_2, "threshold_value"]
                ):
                    remove_list_distance.append(index_2)
                elif (
                    features.loc[index_1, "threshold_value"]
                    < features.loc[index_2, "threshold_value"]
                ):
                    remove_list_distance.append(index_1)
                elif (
                    features.loc[index_1, "threshold_value"]
                    == features.loc[index_2, "threshold_value"]
                ):
                    if features.loc[index_1, "num"] > features.loc[index_2, "num"]:
                        remove_list_distance.append(index_2)
                    elif features.loc[index_1, "num"] < features.loc[index_2, "num"]:
                        remove_list_distance.append(index_1)
                    elif features.loc[index_1, "num"] == features.loc[index_2, "num"]:
                        remove_list_distance.append(index_2)
    features = features[~features.index.isin(remove_list_distance)]
    return features
