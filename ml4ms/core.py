# import json
# import os

# from pathlib import Path
# import re

import numpy as np

# import pandas as pd
# from pymatgen.analysis.bond_valence import BVAnalyzer
# from pymatgen.analysis.local_env import CrystalNN
# from pymatgen.core.structure import Structure
from scipy.interpolate import interp1d

"""
- Helper functions to interpolate XANES and convert collection to a single-property dict
- Define class XANESCollection: list of dictionaries to store data
- functions to process XANES including interpolation
- functions to add target labels
(compute coordination number and charge state using pymatgen)
"""


# Helper function to convert a collection to dict (single property)
def collection_to_single_dict(input_collection, selected_key):
    """
    convert a collection containing attributes 'material_id' and 'selected_key'
    to a dictionary with dict[material_id] = value from input_collection[selected_key]

    :param input_collection: a collection to be converted
    :param selected_key: an attribute from input_collection to be exported to a dict

    :return dict
    """
    # check if selected_key is in input_collection
    if selected_key not in list(input_collection[0].keys()):
        raise Exception("ERROR: Attribute not found in the input collection")

    output_dict = {
        input_collection[i]["material_id"]: input_collection[i][selected_key] for i in range(len(input_collection))
    }
    return output_dict


# Helper function to interpolate XANES spectrum (x,y) onto a new x_domain proj_x
def project_xanes_spectrum(x, y, proj_x, pad: bool = True):
    """
    project (interpolate/extrapolate) a raw (x,y) xanes spectrum
    onto the same grid (per element)

    :param x: x-values (energy) from raw XANES spectra
    :param y: y-values (mu) from raw XANES spectra
    :param proj_x: new x_domains (array)
    :param pad: Pad missing values with 0s. Otherwise extrapolates.

    return projected xanes (y_new)
    """
    # interpolation function
    # Pad with zeroes on the left of the original domain x, (set y_new = 0 where proj_x < min(x)

    xmin = np.min(x)

    if pad:
        x_pad = np.array([proj_x[i] for i in range(len(proj_x)) if proj_x[i] < xmin])
        y_pad = np.array([0.0] * len(x_pad))

        # Return interpolation and extrapolate to right if necessary
        func = interp1d(
            np.concatenate((x_pad, x)),
            np.concatenate((y_pad, y)),
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )
    else:
        func = interp1d(x, y, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    # Occasionally the interpolation returns negative values
    # with the left zero-padding, so manually ceil them to 0
    return list(np.maximum(func(proj_x.round(8)), 0))


class XANESCollection:
    """
    stores XANES data and functions to process XANES + add more attributes to the dataset
    """

    def __init__(
        self,
        xas_collection: list = None,  # collection containing 'material_id' and 'spectrum'
        structure_collection: list = None,  # collection containing 'material_id' and 'structure'
        processed_collection: list = None  # processed collection (optional)
        # that already has both structure and spectra
    ):
        """
        create a XANESCollection instance from an existing raw data collection
        or "processed" data collection

        :param xas_collection: collection downloaded from mpr.materials.xas.search API call,
        must have attributes 'material_id' and 'spectrum' (containing XANES)
        :param structure_collection: collection downloaded from mpr.materials.summary.search API,
        must have attributes 'material_id' and 'structure'
        :param processed_collection: collection (list of dictionaries) containing at least the following:
            'mp_id' : material id
            'xanes': XANES interpolated on a specified 100-pt or 200-pt grid
            (same domains for whole dataset)
            'structure': pymatgen-compatible Structure in a dict format

        """

        # "raw" data [load from raw_spectra_collection or leave blank]
        self.xas_collection = xas_collection if xas_collection else []
        # structure data -> convert to dict for easy use
        self.structure_dict = (
            collection_to_single_dict(structure_collection, "structure") if structure_collection else {}
        )

        # collection with attributes: 'mp_id', 'structure', 'xanes', 'pdf', 'cn', 'cs' etc.
        self.collection = processed_collection if processed_collection else []

        # keep track of what attributes are in the processed collection
        self.collection_attributes = processed_collection[0].keys() if processed_collection else []

        # Pandas DataFrame with columns |mp_id|cn|cs|--XANES--|--PDF--| ready to be split for model training
        self.model_df = None

        # dictionaries containing target labels
        # default format dict[mp_id] = value
        self.cn_dict = {}
        self.cs_dict = {}
        self.cs_mean_dict = {}

    def create_processed_collection(
        self, new_xanes_x_domain, prune_xanes_outliers: bool = True, return_collection: bool = False
    ):
        """
        create a processed_data collection containing 'mp_id', 'xanes' (interpolated XANES), and 'structure'
        :param new_xanes_x_domain: (array) new x domain for interpolation
        :param prune_xanes_outliers: remove entries with xanes spectra outside the
        :param return_collection: (optional) return the processed collection
        else the processed_collection

        :return: (optional) self.collection with structure and interpolated xanes
        """
        # extract lower and upper limits of the new xanes x domain
        lower, upper = min(new_xanes_x_domain), max(new_xanes_x_domain)

        # loop over entries in self.xas_collection
        for item in self.xas_collection:
            curr_mat_id = item["material_id"]
            # only includes item in structure_dict
            if curr_mat_id not in self.structure_dict:
                continue

            # pull structure from dict
            curr_struct = self.structure_dict[curr_mat_id]
            # pull x,y values from xas_collection
            x, y = item["spectrum"]["x"], item["spectrum"]["y"]

            # (optional) skip entries with spectra outside the new x-domain
            # and those with negative y values
            if prune_xanes_outliers:
                ylim_flag = min(y) < 0
                xlim_flag = (max(x) < upper) or (min(x) > lower)
                if ylim_flag or xlim_flag:
                    continue

            # put item + curr_struct + interpolated xanes in a new collection
            curr_dict = {
                "mp_id": curr_mat_id,
                "xanes": project_xanes_spectrum(x, y, new_xanes_x_domain),
                "structure": curr_struct,
            }
            # add to self.collection
            self.collection.append(curr_dict)

        if return_collection:
            return self.collection
