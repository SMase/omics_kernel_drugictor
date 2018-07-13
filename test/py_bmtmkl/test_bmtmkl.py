import sys, os
import unittest
import pandas as pd
import numpy as np
import numpy.matlib


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
		tr_df, _ = self.bmtmkl.get_kernel_df('Drug1')
		ans_df = pd.read_table('./test_data/r_rbf_train.tsv', index_col=0)
		tr_df = tr_df.ix[ans_df.columns, ans_df.columns]

		for ii in range(len(ans_df.columns)):
			np.testing.assert_almost_equal(
				list(tr_df.values[ii]),
				list(ans_df.values[ii]), decimal=18,
				err_msg='error index: ' + str(ii))

	def test_train_bmtmkl_should_be_almost_equal_for_output_of_r_script(self):
		tr_df_list = []
		np.random.seed(1606)
		r_a_mu_train = list(pd.read_table('./test_data/r_a_mu_train.tsv', header=None, index_col=0)[1])
		r_G_mu_train = list(pd.read_table('./test_data/r_G_mu_train.tsv', header=None, index_col=0).T[1])

		for drug_name in self.bmtmkl.drug_list:
			print(drug_name)
			tr_df, _ = self.bmtmkl.get_kernel_df(drug_name)
			if drug_name == 'Drug1':
				ans_df = pd.read_table('./test_data/r_rbf_train.tsv', index_col=0)
				tr_df = tr_df.ix[ans_df.columns, ans_df.columns]
				tr_df_list.append(tr_df)

		log2pi = np.log(2 * np.pi)

		T_len = len(tr_df_list)
		D_mtx = np.zeros([T_len, 1], dtype=int)
		N_mtx = np.zeros([T_len, 1], dtype=int)
		lambda_alpha_list = []
		lambda_beta_list = []
		P = 1

		for ii in range(T_len):
			D_mtx[ii, 0] = tr_df_list[ii].shape[0]
			N_mtx[ii, 0] = tr_df_list[ii].shape[1]

		prm_alpha_lamda = 1
		prm_beta_lamda = 1

		for ii in range(T_len):
			alpha_lamda_mtx = np.full([D_mtx[ii, 0], 1], prm_alpha_lamda + 0.5)
			beta_lamda_mtx = np.full([D_mtx[ii, 0], 1], prm_beta_lamda)
			lambda_alpha_list.append(alpha_lamda_mtx)
			lambda_beta_list.append(beta_lamda_mtx)

		prm_alpha_upsilon = 1
		prm_beta_upsilon = 1

		alpha_upsilon_mtx = np.full([T_len, 1], prm_alpha_upsilon + 0.5 * N_mtx * P)
		beta_upsilon_mtx = np.full([T_len, 1], prm_beta_upsilon)

		a_mu_list = []
		a_sigma_list = []
		G_mu_list = []
		G_sigma_list = []
		for ii in range(T_len):
			a_mu_mtx = np.zeros([D_mtx[ii, 0], 1])
			a_sigma_mtx = np.identity(D_mtx[ii, 0], int)

			for iii in range(D_mtx[ii, 0]):
				a_mu_mtx[iii, 0] = r_a_mu_train[iii]

			a_mu_list.append(a_mu_mtx)
			a_sigma_list.append(a_sigma_mtx)

			G_mu_mtx = np.zeros([P, N_mtx[ii, 0]])
			G_sigma_mtx = np.identity(P, int)

			for iii in range(N_mtx[ii, 0]):
				G_mu_mtx[P-1, iii] = r_G_mu_train[iii]

			G_mu_list.append(G_mu_mtx)
			G_sigma_list.append(G_sigma_mtx)

		prm_alpha_gamma = 1
		prm_beta_gamma = 1
		prm_alpha_omega = 1
		prm_beta_omega = 1
		prm_alpha_epsilon = 1
		prm_beta_epsilon = 1

		alpha_gamma_mtx = np.full([T_len, 1], prm_alpha_gamma + 0.5)
		beta_gamma_mtx = np.full([T_len, 1], prm_beta_gamma)
		alpha_omega_mtx = np.full([P, 1], prm_alpha_omega + 0.5)
		beta_omega_mtx = np.full([P, 1], prm_beta_omega)
		alpha_epsilon_mtx = np.full([T_len, 1], prm_alpha_epsilon + 0.5 * N_mtx)
		beta_epsilon_mtx = np.full([T_len, 1], prm_beta_epsilon)
		be_mu_mtx = np.concatenate([np.zeros([T_len, 1]), np.ones([P, 1])])
		be_sigma_mtx = np.identity(T_len + P, int)

		kmkm_list = []
		for ii in range(T_len):
			kmkm_list.append(np.zeros([D_mtx[ii, 0], D_mtx[ii, 0]]))
			for iii in range(P):
				kmkm_list[ii] = kmkm_list[ii] + np.dot(tr_df_list[ii].values, tr_df_list[ii].T.values)

			tr_df_list[ii] = np.full([D_mtx[ii, 0], N_mtx[ii, 0]*P], tr_df_list[ii])

		prm_progress = 0
		if prm_progress:
			pass

		atimesaT_mu_list = []
		GtimesGT_mu_list = []
		for ii in range(T_len):
			atimesaT_mu_list.append(np.dot(a_mu_list[ii], a_mu_list[ii].T) + a_sigma_list[ii])
			GtimesGT_mu_list.append(np.dot(G_mu_list[ii], G_mu_list[ii].T) + N_mtx[ii] * G_sigma_list[ii])

		btimesbT_mu_mtx = np.dot(be_mu_mtx[0:T_len, 0], be_mu_mtx[0:T_len, 0].T) + be_sigma_mtx[0:T_len, 0:T_len]
		etimeseT_mu_mtx = np.dot(be_mu_mtx[T_len:T_len+P, 0], be_mu_mtx[T_len:T_len+P, 0].T) + be_sigma_mtx[T_len:T_len+P, T_len:T_len+P]
		etimesb_mu_mtx = np.zeros([P, T_len])
		for ii in range(T_len):
			etimesb_mu_mtx[:, ii] = be_mu_mtx[T_len:T_len+P, 0] * be_mu_mtx[ii, 0] + be_sigma_mtx[T_len:T_len+P, ii]

		KmtimesGT_mu_list = []
		for ii in range(T_len):
			KmtimesGT_mu_list.append(np.dot(tr_df_list[ii], numpy.matlib.repmat(G_mu_list[ii].T, P, 1)))

		prm_iter = 200

		for itr in range(prm_iter):

			for ii in range(T_len):
				lambda_beta_list[ii] = 1 / (prm_beta_lamda + 0.5 * np.diag(atimesaT_mu_list[ii]))

			aaa = 0


































































