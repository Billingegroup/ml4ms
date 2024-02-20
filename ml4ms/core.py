# import json
# import os

# from pathlib import Path
# import re

# import numpy as np
# import pandas as pd
# from pymatgen.analysis.bond_valence import BVAnalyzer
# from pymatgen.analysis.local_env import CrystalNN
# from pymatgen.core.structure import Structure
# from scipy.interpolate import interp1d

"""
- Define class XANESCollection: list of dictionaries to store data
- functions to process XANES including interpolation
- functions to add target labels
(compute coordination number and charge state using pymatgen)
"""


class XANESCollection:
    """
    stores XANES data and functions to process XANES + add more attributes to the dataset
    """

    def __init__(
        self,
        xas_collection: list = None,  # collection downloaded from mpr.materials.xas.search
        structure_dict: dict = None,  # dictionary in the format dict[mp_id] = structure
        processed_collection: list = None  # processed collection
        # that already has both structure and spectra
    ):
        """
        create a XANESCollection instance from an existing raw data collection
        or "processed" data collection

        :param xas_collection: collection downloaded from mpr.materials.xas.search API call,
        must have attributes 'material_id' and 'spectrum' (containing XANES)
        :param structure_dict: dictionary in the format dict[mp_id] = structure,
        can be created by manually reformatting the MPDataDocs output of mpr.materials.summary.search
        :param processed_collection: collection (list of dictionaries) containing at least the following:
            'mp_id' : material id
            'xanes': XANES interpolated on a specified 100-pt or 200-pt grid
            (same domains for whole dataset)
            'structure': pymatgen-compatible Structure in a dict format

        """

        # "raw" data [load from raw_spectra_collection or leave blank]
        self.raw_data = xas_collection if xas_collection else []

        # a list of dictionaries with attributes: 'mp_id', 'structure', 'xanes', 'pdf', 'cn', 'cs' etc.
        self.processed_data = processed_collection if processed_collection else []

        # keep track of what attributes are in the processed collection
        self.processed_data_attributes = processed_collection[0].keys() if processed_collection else []

        # Pandas DataFrame with columns |mp_id|cn|cs|--XANES--|--PDF--| ready to be split for model training
        self.model_df = None

        # dictionary containing structure dict
        # default format dict[mp_id] = structure
        self.structure_dict = structure_dict

        # dictionaries containing target labels
        # default format dict[mp_id] = value
        self.cn_dict = {}
        self.cs_dict = {}
        self.cs_mean_dict = {}
