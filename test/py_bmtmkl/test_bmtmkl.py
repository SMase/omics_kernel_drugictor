import sys, os
import unittest
import pandas as pd
import numpy as np
from py_bmtmkl.BMTMKL import BayesianMultitaskMultipleKernelLearning


class TestBayesianMultitaskMultipleKernelLearning(unittest.TestCase):
	def setUp(self):
		self.exp_df = pd.read_table(
			"../data/DREAM7/training data/DREAM7_DrugSensitivity1_GeneExpression.txt", index_col=0)
		self.tr_dr_df = pd.read_table(
			"../data/DREAM7/training data/DREAM7_DrugSensitivity1_Drug_Response_Training.txt", index_col=0)
		self.ts_dr_df = pd.read_table(
			"../data/DREAM7/test data/DREAM7_DrugSensitivity1_test_data.txt", index_col=0)
		pass

	def tearDown(self):
		pass

	def test_rbf_kernel_should_be_equal_for_output_of_r_script(self):
		pass