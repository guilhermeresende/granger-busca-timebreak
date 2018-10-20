cimport numpy as np

cdef float calc_delta(int,int,int, timestamps, deltas) except -1

cdef float P_alpha1(int,int, float, int, float,np.ndarray, np.ndarray, np.ndarray, np.ndarray,timestamps,deltas) except? -1
	
cdef float P_alpha2(int,int, float, int, float,np.ndarray,np.ndarray, np.ndarray, np.ndarray,timestamps,deltas) except? -1

cdef float P_k(float,int, float,np.ndarray,np.ndarray, np.ndarray, np.ndarray,timestamps,deltas) except? -1	

cdef float P_mu1(int,float,int, float, np.ndarray,np.ndarray, np.ndarray, np.ndarray, timestamps, deltas) except? -1

cdef float P_mu2(int,float,int, float, np.ndarray,np.ndarray,np.ndarray, np.ndarray, timestamps, deltas) except? -1

cdef float calc_ratio(float prob)