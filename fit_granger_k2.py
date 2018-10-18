import numpy as np
from bisect import bisect
import scipy.stats as ss
import matplotlib.pyplot as plt
from time import time
class Granger_k():
	def __init__(self,timestamps,T=0):
		self.timestamps=timestamps
		self.d=len(timestamps)
		if(T==0):
			self.T=max([t for timestamp in timestamps for t in timestamp])
		else:
			self.T=T
		self.deltas={}

	def calc_delta(self,p1,p2,t_idx):
		str_delta=str(p1)+'_'+str(p2)+'_'+str(t_idx)
		if (str_delta in self.deltas):
			return self.deltas[str_delta]
		tp=self.timestamps[p1][t_idx]
		tpp_idx = bisect(self.timestamps[p2], tp)
		if tpp_idx == len(self.timestamps[p2]):
			tpp_idx -= 1
		tpp = self.timestamps[p2][tpp_idx]
		while tpp >= tp and tpp_idx > 0:
			tpp_idx -= 1
			tpp = self.timestamps[p2][tpp_idx]
		if tpp >= tp:
			return 0

		self.deltas[str_delta]=(tp-tpp)
		return tp - tpp

	def calc_ratio(self,prob):
		try:
			r=np.exp(prob)
		except FloatingPointError as err:
			if prob<0:
				return 0
			else:
				return 1
		return r

	def P_k(self,k,Alpha1,Alpha2,Mu):	
		if(k<0 or k>self.T):
			return float("-inf")
		
		p=0
		for p1 in range(self.d):
			k_idx=len(self.timestamps[p1]) -1 
			delta_ba=0
			past_term=[0]*self.d
			for t_idx in range(len(self.timestamps[p1])):
				first_term1=0
				second_term1=0
				first_term2=0
				second_term2=0
				
				if(self.timestamps[p1][t_idx]<=k):
					for p2 in range(self.d):
						delta_ba=self.calc_delta(p1,p2,t_idx)				
						if(delta_ba>0):
							first_term1+=Alpha1[p2][p1]/(1.+delta_ba)
						
						first_term2+=past_term[p2]*(self.timestamps[p1][t_idx]-self.timestamps[p1][t_idx-1])
						
						if(delta_ba>0):
							past_term[p2]=Alpha1[p2][p1]/(1.+delta_ba)					
						else:
							past_term[p2]=0.				
							
					p+=np.log(Mu[p1]+first_term1)-first_term2
				else:
					for p2 in range(self.d):
							
						delta_ba=self.calc_delta(p1,p2,t_idx)				
						if(delta_ba>0):
							second_term1+=Alpha2[p2][p1]/(1.+delta_ba)
							
						second_term2+=past_term[p2]*(self.timestamps[p1][t_idx]-self.timestamps[p1][t_idx-1])
						
						if(delta_ba>0):
							past_term[p2]=Alpha2[p2][p1]/(1.+delta_ba)					
						else:
							past_term[p2]=0.
							
					p+=np.log(Mu[p1]+second_term1)-second_term2
			second_term2=0
			for p2 in range(self.d):
				if(t_idx>0):
					second_term2+=past_term[p2]*(self.T-self.timestamps[p1][t_idx])
		
			p-=second_term2
		return p


	def P_mu(self,p1,k,Alpha1,Alpha2,Mu):
		if(Mu[p1]<=0):
			return float("-inf")
		
		p=0	
		for t_idx in range(len(self.timestamps[p1])):
				first_term1=0
				second_term1=0
				
				if(self.timestamps[p1][t_idx]<=k):
					for p2 in range(self.d):
						delta_ba=self.calc_delta(p1,p2,t_idx)				
						if(delta_ba>0):
							first_term1+=Alpha1[p2][p1]/(1.+delta_ba)				   
					p+=np.log(Mu[p1]+first_term1)
				else:
					for p2 in range(self.d):						
						delta_ba=self.calc_delta(p1,p2,t_idx)				
						if(delta_ba>0):
							second_term1+=Alpha2[p2][p1]/(1.+delta_ba)
							
					p+=np.log(Mu[p1]+second_term1)
					
		p-=self.T*(Mu[p1])
		return p

	def P_alpha1(self,p1,p2,k,Alpha1,Alpha2,Mu):
		
		if(Alpha1[p2][p1]<0):
			return float("-inf")
		
		p=0
		k_idx=len(self.timestamps[p1]) -1 
		delta_ba=0
		past_term=[0]*self.d
		last_t_idx=0
		for t_idx in range(len(self.timestamps[p1])):
				first_term1=0
				first_term2=0
				if(self.timestamps[p1][t_idx]<=k):
					last_t_idx=t_idx
					for p2_i in range(self.d):
						delta_ba=self.calc_delta(p1,p2_i,t_idx)				
						if(delta_ba>0):
							first_term1+=Alpha1[p2_i][p1]/(1.+delta_ba)
							#print(self.timestamps[p1][t_idx],p1,p2_i,Alpha1[p2_i][p1])
						if(p2==p2_i):
							first_term2+=past_term[p2_i]*(self.timestamps[p1][t_idx]-self.timestamps[p1][t_idx-1])
						
						if(delta_ba>0):
							past_term[p2_i]=Alpha1[p2_i][p1]/(1.+delta_ba)					
						else:
							past_term[p2_i]=0.  
					p+=np.log(Mu[p1]+first_term1)-first_term2
		
		first_term2=0
		
		if (last_t_idx==(len(self.timestamps[p1])-1)):
			final_t=self.T
		else:
			final_t=self.timestamps[p1][last_t_idx+1]
		delta_ba=self.calc_delta(p1,p2,last_t_idx)	   
		if(last_t_idx>0 and delta_ba>0):
			first_term2=Alpha1[p2][p1]/(1.+delta_ba)*(final_t-self.timestamps[p1][last_t_idx])
		p-=first_term2
		return p

	def P_alpha2(self,p1,p2,k,Alpha1,Alpha2,Mu):
		
		if(Alpha2[p2][p1]<0):
			return float("-inf")  
		p=0
		k_idx=len(self.timestamps[p1]) -1 
		delta_ba=0
		past_term=[0]*self.d
		for t_idx in range(len(self.timestamps[p1])):
				second_term1=0
				second_term2=0
				if(self.timestamps[p1][t_idx]>k):
					for p2_i in range(self.d):
						delta_ba=self.calc_delta(p1,p2_i,t_idx)				
						if(delta_ba>0):
							second_term1+=Alpha2[p2_i][p1]/(1.+delta_ba)
						if(p2==p2_i):
							second_term2+=past_term[p2_i]*(self.timestamps[p1][t_idx]-self.timestamps[p1][t_idx-1])
						
						if(delta_ba>0):
							past_term[p2_i]=Alpha2[p2_i][p1]/(1.+delta_ba)   
						else:
							past_term[p2_i]=0.										
							
					p+=np.log(Mu[p1]+second_term1)-second_term2
						
		second_term2=0	

		if not (self.timestamps[p1][-1]<=k or self.timestamps[p1][-2]<=k):
			delta_ba=self.calc_delta(p1,p2,t_idx)	
			if(t_idx>0 and delta_ba>0):
				second_term2=past_term[p2]*(self.T-self.timestamps[p1][t_idx])

		p-=second_term2
		return p
				
	def fit(self,n_iter=400,var_k=15,var_mu=0.03,var_alpha=0.08,burn_in=150):
		self.n_iter=n_iter
		self.burn_in=burn_in
		initial_value=1.0/self.d
		initial_time=600.
		self.k_v=[initial_time]
		self.Mu_v=[np.ones(shape=(self.d))*initial_value]
		self.Alpha1_v=[np.ones(shape=(self.d,self.d))*initial_value]
		self.Alpha2_v=[np.ones(shape=(self.d,self.d))*initial_value]

		k=self.k_v[0]
		Mu=self.Mu_v[0]
		Alpha1=self.Alpha1_v[0]
		Alpha2=self.Alpha2_v[0]

		self.reject_k=0.
		self.reject_mu=[0.]*self.d
		self.reject_alpha1=[[0.]*self.d for i in range(self.d)]
		self.reject_alpha2=[[0.]*self.d for i in range(self.d)]

		for i in range(self.n_iter):
			t_begin=time()
			######## k ########
			past_p_k=self.P_k(self.k_v[-1],self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu_v[-1])
			new_k=ss.norm.rvs(self.k_v[-1],var_k)
			new_p_k=self.P_k(new_k,self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu_v[-1])
			
			r=self.calc_ratio(new_p_k - past_p_k)
			
			if(r>=1 or ss.uniform.rvs()<r):
				print("Accepted",new_k)
				self.k_v.append(new_k)
			else:
				self.k_v.append(self.k_v[-1])
				self.reject_k+=1
			
			k=self.k_v[-1]
			########  ########

			new_Mu=np.copy(self.Mu_v[-1])
			new_Alpha1=np.copy(self.Alpha1_v[-1])
			new_Alpha2=np.copy(self.Alpha2_v[-1])
			for p1 in range(self.d):
				######## Mu ########
				past_p_Mu=self.P_mu(p1,k,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu)
				new_Mu[p1]=ss.norm.rvs(self.Mu_v[-1][p1],scale=var_mu)
				new_p_Mu=self.P_mu(p1,k,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu)
				
				r=self.calc_ratio(new_p_Mu-past_p_Mu)

				if(r>=1 or ss.uniform.rvs()<r):
					pass
				else:
					new_Mu[p1]=self.Mu_v[-1][p1]
					self.reject_mu[p1]+=1
				##########  ########
				
				for p2 in range(self.d):

					##### Alpha1 #####
					past_p_Alpha1=self.P_alpha1(p1,p2,k,new_Alpha1,new_Alpha2,new_Mu)
					new_Alpha1[p2][p1]=ss.norm.rvs(self.Alpha1_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha1=self.P_alpha1(p1,p2,k,new_Alpha1,new_Alpha2,new_Mu)
					r=self.calc_ratio(new_p_Alpha1- past_p_Alpha1)	 
					if(r>=1 or ss.uniform.rvs()<r):
						#past_p_Alpha1[p1][p2]=new_p_Alpha1  
						pass
					else:
						new_Alpha1[p2][p1]=self.Alpha1_v[-1][p2][p1]
						self.reject_alpha1[p2][p1]+=1
						
					##### Alpha2 #####
					past_p_Alpha2=self.P_alpha2(p1,p2,k,new_Alpha1,new_Alpha2,new_Mu)
					new_Alpha2[p2][p1]=ss.norm.rvs(self.Alpha2_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha2=self.P_alpha2(p1,p2,k,new_Alpha1,new_Alpha2,new_Mu)
				
					r=self.calc_ratio(new_p_Alpha2- past_p_Alpha2)
					if(r>=1 or ss.uniform.rvs()<r):
						#past_p_Alpha2[p1][p2]=new_p_Alpha2
						pass
					else:
						new_Alpha2[p2][p1]=self.Alpha2_v[-1][p2][p1]
						self.reject_alpha2[p2][p1]+=1
			
			self.Mu_v.append(new_Mu)
			self.Alpha1_v.append(new_Alpha1)
			self.Alpha2_v.append(new_Alpha2)

			print("iteration",i,"time",time()-t_begin)
			
			
		self.k_v=self.k_v[self.burn_in:]
		self.Mu_v=self.Mu_v[self.burn_in:]
		self.Alpha1_v=self.Alpha1_v[self.burn_in:]
		self.Alpha2_v=self.Alpha2_v[self.burn_in:]

		print("k",np.mean(self.k_v))
		for p1 in range(self.d):
			print("mu",p1,np.mean([self.Mu_v[i][p1] for i in range(n_iter-burn_in)]))
		for p1 in range(self.d):
			for p2 in range(self.d):
				print("Alpha1",p1,p2,np.mean([self.Alpha1_v[i][p1][p2] for i in range(n_iter-burn_in)]))
		for p1 in range(self.d):
			for p2 in range(self.d):
				print("Alpha2",p1,p2,np.mean([self.Alpha2_v[i][p1][p2] for i in range(n_iter-burn_in)]))

		return((self.k_v,self.Mu_v,self.Alpha1_v,self.Alpha2_v))
				
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
