import pandas as pd
import numpy as np


class BayesianMultitaskMultipleKernelLearning:
	def __init__(self, ):
		self.exp_df = pd.read_table("../data/DREAM7/training data/DREAM7_DrugSensitivity1_GeneExpression.txt", index_col=0)
		self.tr_dr_df = pd.read_table("../data/DREAM7/training data/DREAM7_DrugSensitivity1_Drug_Response_Training.txt",
		                              index_col=0)
		self.ts_dr_df = pd.read_table("../data/DREAM7/test data/DREAM7_DrugSensitivity1_test_data.txt", index_col=0)

	def get_matlix_rdf_kernel_dot(self, df1, df2, gamma):
		diff = np.zeros((len(df1.columns), len(df2.columns)))
		for i in range(len(df1.columns)):
			for j in range(len(df2.columns)):
				diff[i, j] = np.sum((df1.values[:, i] - df2.values[:, j]) ** 2)

		kernel_df = pd.DataFrame(np.exp(-diff * gamma))
		kernel_df.index = df1.columns
		kernel_df.columns = df2.columns
		return kernel_df

	def get_kernel_list(self):
		tr_dict = {}
		ts_dict = {}
		drug_list = [drug_name for drug_name in self.tr_dr_df.columns if drug_name in self.ts_dr_df.columns]
		print(drug_list)
		for name in drug_list:
			tr_dr_cellline_index = self.tr_dr_df[name].dropna(how='all').index
			ts_dr_cellline_index = self.ts_dr_df[name].dropna(how='all').index

			tr_cell_line_name = [name for name in tr_dr_cellline_index if name in list(self.exp_df.columns)]
			ts_cell_line_name = [name for name in ts_dr_cellline_index if name in list(self.exp_df.columns)]

			omics_tr_df = self.exp_df[tr_cell_line_name]
			omics_ts_df = self.exp_df[ts_cell_line_name]

			tr_dict[name] = self.get_matlix_rdf_kernel_dot(omics_tr_df, omics_tr_df, 0.001)
			ts_dict[name] = self.get_matlix_rdf_kernel_dot(omics_ts_df, omics_tr_df, 0.001)

		return tr_dict, ts_dict


