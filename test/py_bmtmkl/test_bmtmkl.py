import sys, os
import unittest
import pandas as pd
import numpy as np
from py_bmtmkl.BMTMKL import BayesianMultitaskMultipleKernelLearning


class TestBayesianMultitaskMultipleKernelLearning(unittest.TestCase):
	def setUp(self):
		self.bmtmkl = BayesianMultitaskMultipleKernelLearning(
			exp_path="/Users/shogo_mase/PycharmProjects/Kernelized-Rank-Learning/data/DREAM7/training data/DREAM7_DrugSensitivity1_GeneExpression.txt",
			tr_dr_path="/Users/shogo_mase/PycharmProjects/Kernelized-Rank-Learning/data/DREAM7/training data/DREAM7_DrugSensitivity1_Drug_Response_Training.txt",
			ts_dr_path="/Users/shogo_mase/PycharmProjects/Kernelized-Rank-Learning/data/DREAM7/test data/DREAM7_DrugSensitivity1_test_data.txt",
		)

	def tearDown(self):
		pass

	def test_rbf_kernel_should_be_almost_equal_for_output_of_r_script(self):
		# significant figure 18 is OK, but 19 is NG
		tr_df_dict, _ = self.bmtmkl.get_kernel_dict()
		ans_df = pd.read_table('./test_data/r_rbf_train.tsv', index_col=0)
		tr_df_dict['Drug1'] = tr_df_dict['Drug1'].ix[ans_df.columns, ans_df.columns]

		for ii in range(len(ans_df.columns)):
			np.testing.assert_almost_equal(
				list(tr_df_dict['Drug1'].values[ii]),
				list(ans_df.values[ii]), decimal=18,
				err_msg='error index: ' + str(ii))


