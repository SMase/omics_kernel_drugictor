import pandas as pd
import numpy as np
from collections import OrderedDict

class BayesianMultitaskMultipleKernelLearning:
	def __init__(self, exp_path, tr_dr_path, ts_dr_path):
		self._exp_df = pd.read_table(exp_path, index_col=0)
		self._tr_dr_df = pd.read_table(tr_dr_path, index_col=0)
		self._ts_dr_df = pd.read_table(ts_dr_path, index_col=0)
		self._drug_list = [drug_name for drug_name in self._tr_dr_df.columns if drug_name in self._ts_dr_df.columns]

	@property
	def drug_list(self):
		return self._drug_list

	def get_matlix_rdf_kernel_dot(self, df1, df2, gamma):
		diff = np.zeros((len(df1.columns), len(df2.columns)))
		for i in range(len(df1.columns)):
			for j in range(len(df2.columns)):
				diff[i, j] = np.sum((df1.values[:, i] - df2.values[:, j]) ** 2)

		kernel_df = pd.DataFrame(np.exp(-diff * gamma))
		kernel_df.index = df1.columns
		kernel_df.columns = df2.columns
		return kernel_df

	def get_kernel_df(self, drug_name):
		tr_dr_cellline_index = self._tr_dr_df[drug_name].dropna(how='all').index
		ts_dr_cellline_index = self._ts_dr_df[drug_name].dropna(how='all').index

		tr_cell_line_name = [name for name in tr_dr_cellline_index if name in list(self._exp_df.columns)]
		ts_cell_line_name = [name for name in ts_dr_cellline_index if name in list(self._exp_df.columns)]

		omics_tr_df = self._exp_df[tr_cell_line_name]
		omics_ts_df = self._exp_df[ts_cell_line_name]

		tr_kernel_df = self.get_matlix_rdf_kernel_dot(omics_tr_df, omics_tr_df, 0.001)
		ts_kernel_df = self.get_matlix_rdf_kernel_dot(omics_ts_df, omics_tr_df, 0.001)

		return tr_kernel_df, ts_kernel_df













