import numpy as np
cimport numpy as np
from bisect import bisect


cdef float calc_delta(int p1,int p2,int t_idx, timestamps, deltas) except -1:
	cdef int tpp_idx
	cdef float tp,tpp
	str_delta = str(p1)+'_'+str(p2)+'_'+str(t_idx)
	if (str_delta in deltas):
		return deltas[str_delta]
	tp = timestamps[p1][t_idx]
	tpp_idx = bisect(timestamps[p2], tp)
	if tpp_idx == len(timestamps[p2]):
		tpp_idx -= 1
	tpp = timestamps[p2][tpp_idx]
	while tpp >= tp and tpp_idx > 0:
		tpp_idx -= 1
		tpp = timestamps[p2][tpp_idx]
	if tpp >= tp:
		return 0

	deltas[str_delta]=(tp-tpp)
	return tp - tpp

cdef float P_k(float k,int d, float T,np.ndarray Alpha1,np.ndarray Alpha2, np.ndarray Mu1, np.ndarray Mu2,timestamps,deltas) except? -1:	
	if(k<0 or k>T):
		return float("-inf")
	
	cdef float p=0
	cdef int k_idx
	cdef float delta_ba
	cdef float first_term1,first_term2,second_term1,second_term2

	for p1 in range(d):
		k_idx=len(timestamps[p1]) -1 
		delta_ba=0
		past_term=[0]*d
		for t_idx in range(len(timestamps[p1])):
			first_term1=0
			second_term1=0
			first_term2=0
			second_term2=0
			
			if(timestamps[p1][t_idx]<=k):
				for p2 in range(d):
					delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)				
					if(delta_ba>0):
						first_term1+=Alpha1[p2][p1]/(1.+delta_ba)
					
					first_term2+=past_term[p2]*(timestamps[p1][t_idx]-timestamps[p1][t_idx-1])
					
					if(delta_ba>0):
						past_term[p2]=Alpha1[p2][p1]/(1.+delta_ba)					
					else:
						past_term[p2]=0.				
						
				p+=np.log(Mu1[p1]+first_term1)-first_term2
			else:
				for p2 in range(d):
						
					delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)			
					if(delta_ba>0):
						second_term1+=Alpha2[p2][p1]/(1.+delta_ba)
						
					second_term2+=past_term[p2]*(timestamps[p1][t_idx]-timestamps[p1][t_idx-1])
					
					if(delta_ba>0):
						past_term[p2]=Alpha2[p2][p1]/(1.+delta_ba)					
					else:
						past_term[p2]=0.
						
				p+=np.log(Mu2[p1]+second_term1)-second_term2
		second_term2=0
		for p2 in range(d):
			if(t_idx>0):
				second_term2+=past_term[p2]*(T-timestamps[p1][t_idx])
	
		p-=second_term2
		p-=k*(Mu1[p1])+(T-k)*(Mu2[p1])

	return p

cdef float P_mu1(int p1,float k,int d, float T, np.ndarray Alpha1,np.ndarray Alpha2, np.ndarray Mu1, np.ndarray Mu2, timestamps, deltas) except? -1:
	if(Mu1[p1]<=0):
		return float("-inf")
	
	cdef float p=0	
	cdef float first_term1, second_term1
	for t_idx in range(len(timestamps[p1])):
			first_term1=0

			if(timestamps[p1][t_idx]<=k):
				for p2 in range(d):
					delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)			
					if(delta_ba>0):
						first_term1+=Alpha1[p2][p1]/(1.+delta_ba)				   
				p+=np.log(Mu1[p1]+first_term1)
						
				
	p-=k*(Mu1[p1])
	return p

cdef float P_mu2(int p1,float k,int d, float T, np.ndarray Alpha1,np.ndarray Alpha2,np.ndarray Mu1, np.ndarray Mu2, timestamps, deltas) except? -1:
	if(Mu2[p1]<=0):
		return float("-inf")
	
	cdef float p=0	
	cdef float second_term1
	for t_idx in range(len(timestamps[p1])):
			second_term1=0
			
			if(timestamps[p1][t_idx]>k):
				for p2 in range(d):
					delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)			
					if(delta_ba>0):
						second_term1+=Alpha2[p2][p1]/(1.+delta_ba)
						
				p+=np.log(Mu2[p1]+second_term1)
				
	p-=(T-k)*(Mu2[p1])
	return p


cdef float P_alpha1(int p1,int p2, float k, int d, float T,np.ndarray Alpha1,np.ndarray Alpha2, np.ndarray Mu1, np.ndarray Mu2,timestamps,deltas) except? -1:
	if(Alpha1[p2][p1]<0):
		return float("-inf")
	
	cdef float p=0
	cdef int k_idx=len(timestamps[p1]) -1 
	cdef float delta_ba=0
	past_term=[0]*d
	cdef int last_t_idx=0
	cdef first_term1,first_term2
	for t_idx in range(len(timestamps[p1])):
			first_term1=0
			first_term2=0
			if(timestamps[p1][t_idx]<=k):
				last_t_idx=t_idx
				for p2_i in range(d):
					delta_ba=calc_delta(p1,p2_i,t_idx,timestamps,deltas)				
					if(delta_ba>0):
						first_term1+=Alpha1[p2_i][p1]/(1.+delta_ba)
					if(p2==p2_i):
						first_term2+=past_term[p2_i]*(timestamps[p1][t_idx]-timestamps[p1][t_idx-1])
					
					if(delta_ba>0):
						past_term[p2_i]=Alpha1[p2_i][p1]/(1.+delta_ba)					
					else:
						past_term[p2_i]=0.  
				p+=np.log(Mu1[p1]+first_term1)-first_term2
	
	first_term2=0
	
	if (last_t_idx==(len(timestamps[p1])-1)):
		final_t=T
	else:
		final_t=timestamps[p1][last_t_idx+1]
	delta_ba=calc_delta(p1,p2,last_t_idx,timestamps,deltas)	   
	if(last_t_idx>0 and delta_ba>0):
		first_term2=Alpha1[p2][p1]/(1.+delta_ba)*(final_t-timestamps[p1][last_t_idx])
	p-=first_term2
	return p
		
cdef float P_alpha2(int p1,int p2, float k, int d, float T,np.ndarray Alpha1,np.ndarray Alpha2, np.ndarray Mu1, np.ndarray Mu2,timestamps,deltas) except? -1:
	if(Alpha2[p2][p1]<0):
		return float("-inf")  
	cdef float p=0
	cdef int k_idx=len(timestamps[p1]) -1 
	cdef float delta_ba=0
	past_term=[0]*d
	cdef int last_t_idx=0
	cdef second_term1,second_term2
	for t_idx in range(len(timestamps[p1])):
			second_term1=0
			second_term2=0
			if(timestamps[p1][t_idx]>k):
				for p2_i in range(d):
					delta_ba=calc_delta(p1,p2_i,t_idx,timestamps,deltas)			
					if(delta_ba>0):
						second_term1+=Alpha2[p2_i][p1]/(1.+delta_ba)
					if(p2==p2_i):
						second_term2+=past_term[p2_i]*(timestamps[p1][t_idx]-timestamps[p1][t_idx-1])
					
					if(delta_ba>0):
						past_term[p2_i]=Alpha2[p2_i][p1]/(1.+delta_ba)   
					else:
						past_term[p2_i]=0.										
						
				p+=np.log(Mu2[p1]+second_term1)-second_term2
					
	second_term2=0	

	if (len(timestamps[p1])>1) and not (timestamps[p1][-1]<=k or timestamps[p1][-2]<=k):
		delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)	
		if(t_idx>0 and delta_ba>0):
			second_term2=past_term[p2]*(T-timestamps[p1][t_idx])

	p-=second_term2
	return p



cdef float calc_ratio(float prob):
	try:
		r=np.exp(prob)
	except FloatingPointError as err:
		if prob<0:
			return 0
		else:
			return 1
	return r
