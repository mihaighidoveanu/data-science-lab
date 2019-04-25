class OutlierIQR(object):
	"""docstring for OutlierIQR"""
	def __init__(self):
		super(OutlierIQR, self).__init__()
		

	def fit(self, X, columns = None):
		if columns is None:
			cX = X
		else:
			cX = X[columns]
		self.Q1 = cX.quantile(.25)
		self.Q3 = cX.quantile(.75)

	def transform(self, X):
		Q1 = self.Q1
		Q3 = self.Q3
		IQR = Q3 - Q1
		mask = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
		X = X[~mask.any(axis = 1)]
		return X

