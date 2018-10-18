import numpy as np
from bisect import bisect
import scipy.stats as ss
import matplotlib.pyplot as plt
from time import time
cimport numpy as np


class Granger_k():
	def __init__(self,timestamps,T=0):
		self.timestamps=timestamps
		self.d=len(timestamps)
		if(T==0):
			self.T=max([t for timestamp in timestamps for t in timestamp])
		else:
			self.T=T
		self.deltas={}

	def calc_ratio(self,prob):
		try:
			r=np.exp(prob)
		except FloatingPointError as err:
			if prob<0:
				return 0
			else:
				return 1
		return r
				
	def fit(self,n_iter=400,var_k=15,var_mu=0.03,var_alpha=0.08,burn_in=150,initial_time=0):
		initial_time=self.T/2.0 if initial_time==0 else initial_time
		self.n_iter=n_iter
		self.burn_in=burn_in
		initial_value=1.0/self.d		
		self.k_v=[initial_time]
		self.Mu1_v=[np.ones(shape=(self.d))*initial_value]
		self.Mu2_v=[np.ones(shape=(self.d))*initial_value]
		self.Alpha1_v=[np.ones(shape=(self.d,self.d))*initial_value]
		self.Alpha2_v=[np.ones(shape=(self.d,self.d))*initial_value]

		k=self.k_v[0]
		Mu1=self.Mu1_v[0]
		Mu2=self.Mu2_v[0]
		Alpha1=self.Alpha1_v[0]
		Alpha2=self.Alpha2_v[0]

		self.reject_k=0.
		self.reject_mu1=[0.]*self.d
		self.reject_mu2=[0.]*self.d
		self.reject_alpha1=[[0.]*self.d for i in range(self.d)]
		self.reject_alpha2=[[0.]*self.d for i in range(self.d)]

		for i in range(self.n_iter):
			t_begin=time()
			######## k ########
			past_p_k=P_k(self.k_v[-1],self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu1_v[-1],self.Mu2_v[-1],self.timestamps,self.deltas)
			new_k=ss.norm.rvs(self.k_v[-1],var_k)
			new_p_k=P_k(new_k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu1_v[-1],self.Mu2_v[-1],self.timestamps,self.deltas)
			
			r=calc_ratio(new_p_k - past_p_k)
			
			if(r>=1 or ss.uniform.rvs()<r):
				print("Accepted",new_k)
				self.k_v.append(new_k)
			else:
				self.k_v.append(self.k_v[-1])
				self.reject_k+=1
			
			k=self.k_v[-1]
			########  ########

			new_Mu1=np.copy(self.Mu1_v[-1])
			new_Mu2=np.copy(self.Mu2_v[-1])
			new_Alpha1=np.copy(self.Alpha1_v[-1])
			new_Alpha2=np.copy(self.Alpha2_v[-1])
			for p1 in range(self.d):
				######## Mu1 ########
				past_p_Mu1=P_mu1(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				new_Mu1[p1]=ss.norm.rvs(self.Mu1_v[-1][p1],scale=var_mu)
				new_p_Mu1=P_mu1(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
				r=calc_ratio(new_p_Mu1-past_p_Mu1)

				if(r>=1 or ss.uniform.rvs()<r):
					pass
				else:
					new_Mu1[p1]=self.Mu1_v[-1][p1]
					self.reject_mu1[p1]+=1
				##########  ########
				
				######## Mu2 ########
				past_p_Mu2=P_mu2(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				new_Mu2[p1]=ss.norm.rvs(self.Mu1_v[-1][p1],scale=var_mu)
				new_p_Mu2=P_mu2(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
				r=calc_ratio(new_p_Mu2-past_p_Mu2)

				if(r>=1 or ss.uniform.rvs()<r):
					pass
				else:
					new_Mu2[p1]=self.Mu2_v[-1][p1]
					self.reject_mu2[p1]+=1
				##########  ########

				for p2 in range(self.d):

					##### Alpha1 #####
					past_p_Alpha1=P_alpha1(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					new_Alpha1[p2][p1]=ss.norm.rvs(self.Alpha1_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha1=P_alpha1(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					r=calc_ratio(new_p_Alpha1- past_p_Alpha1)	 
					if(r>=1 or ss.uniform.rvs()<r):
						#past_p_Alpha1[p1][p2]=new_p_Alpha1  
						pass
					else:
						new_Alpha1[p2][p1]=self.Alpha1_v[-1][p2][p1]
						self.reject_alpha1[p2][p1]+=1
						
					##### Alpha2 #####
					past_p_Alpha2=P_alpha2(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					new_Alpha2[p2][p1]=ss.norm.rvs(self.Alpha2_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha2=P_alpha2(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
					r=calc_ratio(new_p_Alpha2- past_p_Alpha2)
					if(r>=1 or ss.uniform.rvs()<r):
						#past_p_Alpha2[p1][p2]=new_p_Alpha2
						pass
					else:
						new_Alpha2[p2][p1]=self.Alpha2_v[-1][p2][p1]
						self.reject_alpha2[p2][p1]+=1
			
			self.Mu1_v.append(new_Mu1)
			self.Mu2_v.append(new_Mu2)
			self.Alpha1_v.append(new_Alpha1)
			self.Alpha2_v.append(new_Alpha2)

			print("iteration",i,"time",time()-t_begin)
			
			
		self.k_v=self.k_v[self.burn_in:]
		self.Mu1_v=self.Mu1_v[self.burn_in:]
		self.Mu2_v=self.Mu2_v[self.burn_in:]
		self.Alpha1_v=self.Alpha1_v[self.burn_in:]
		self.Alpha2_v=self.Alpha2_v[self.burn_in:]

		print("k",np.mean(self.k_v))
		for p1 in range(self.d):
			print("mu1",p1,np.mean([self.Mu1_v[i][p1] for i in range(n_iter-burn_in)]))
		for p1 in range(self.d):
			print("mu2",p1,np.mean([self.Mu2_v[i][p1] for i in range(n_iter-burn_in)]))
		for p1 in range(self.d):
			for p2 in range(self.d):
				print("Alpha1",p1,p2,np.mean([self.Alpha1_v[i][p1][p2] for i in range(n_iter-burn_in)]))
		for p1 in range(self.d):
			for p2 in range(self.d):
				print("Alpha2",p1,p2,np.mean([self.Alpha2_v[i][p1][p2] for i in range(n_iter-burn_in)]))

		return((self.k_v,self.Mu1_v,self.Mu2_v,self.Alpha1_v,self.Alpha2_v))
				
	def print_rejection_rate(self):
		print("k",self.reject_k/self.n_iter)
		print("mu",[i/self.n_iter for i in self.reject_mu])
		print("alpha1",[[i/self.n_iter for i in j] for j in self.reject_alpha1])
		print("alpha2",[[i/self.n_iter for i in j] for j in self.reject_alpha2])

	def plot_values(self):		
		plt.plot(range(len(self.k_v)),self.k_v)
		plt.title('k')
		plt.show()

		Mu_lista=[[] for i in range(self.d)]
		for i in range(self.d):
			for i_iter in range(0,self.n_iter-self.burn_in):
				self.Mu_v[i_iter]
				Mu_lista[i].append(self.Mu_v[i_iter][i])
		for i in range(self.d):
			plt.plot(range(len(Mu_lista[i])),Mu_lista[i],label='Mu '+str(i))
		plt.legend()
		plt.show()  

		Alpha1_lista=[]
		Alpha2_lista=[]
		for i in range(self.d):
			Alpha1_lista.append([[] for i in range(self.d)])
			Alpha2_lista.append([[] for i in range(self.d)])
		for i in range(self.d):
			for j in range(self.d):
				for i_iter in range(0,self.n_iter-self.burn_in):
					Alpha1_lista[i][j].append(self.Alpha1_v[i_iter][i][j])
					Alpha2_lista[i][j].append(self.Alpha2_v[i_iter][i][j])
		for i in range(self.d):
			for j in range(self.d):
				plt.plot(range(len(Alpha1_lista[i][j])),Alpha1_lista[j][i],label='Alpha1 %d %d' % (j,i))
				
			plt.legend()
			plt.show()  
			
		for i in range(self.d):
			for j in range(self.d):
				plt.plot(range(len(Alpha2_lista[i][j])),Alpha2_lista[j][i],label='Alpha2 %d %d' % (j,i))
				
			plt.legend()
			plt.show() 

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
							#print(timestamps[p1][t_idx],p1,p2_i,Alpha1[p2_i][p1])
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
					#delta_ba=calc_delta(p1,p2_i,t_idx)	
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
		#delta_ba=calc_delta(p1,p2,t_idx)
		delta_ba=calc_delta(p1,p2,t_idx,timestamps,deltas)	
		if(t_idx>0 and delta_ba>0):
			second_term2=past_term[p2]*(T-timestamps[p1][t_idx])

	p-=second_term2
	return p

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
						#delta_ba=calc_delta(p1,p2,t_idx)
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
							
						#delta_ba=calc_delta(p1,p2,t_idx)
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
						#delta_ba=calc_delta(p1,p2,t_idx)				
						if(delta_ba>0):
							first_term1+=Alpha1[p2][p1]/(1.+delta_ba)				   
					p+=np.log(Mu1[p1]+first_term1)
							
					
		p-=k*(Mu1[p1])
		return p

cdef float P_mu2(int p1,float k,int d, float T, np.ndarray Alpha1,np.ndarray Alpha2,np.ndarray Mu1, np.ndarray Mu2, timestamps, deltas) except? -1:
		if(Mu2[p1]<=0):
			return float("-inf")
		
		cdef float p=0	
		cdef float first_term1, second_term1
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

cdef float calc_ratio(float prob):
	try:
		r=np.exp(prob)
	except FloatingPointError as err:
		if prob<0:
			return 0
		else:
			return 1
	return r
