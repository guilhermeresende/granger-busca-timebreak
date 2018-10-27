import numpy as np
from bisect import bisect
import scipy.stats as ss
import matplotlib.pyplot as plt
from time import time
cimport likelihood_calc

class Granger_k():
	def __init__(self,timestamps,T=0):
		self.timestamps=timestamps
		self.d=len(timestamps)
		if(T==0):
			self.T=max([t for timestamp in timestamps for t in timestamp])
		else:
			self.T=T
		self.deltas={}

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
			past_p_k=likelihood_calc.P_k(self.k_v[-1],self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu1_v[-1],self.Mu2_v[-1],self.timestamps,self.deltas)
			new_k=ss.norm.rvs(self.k_v[-1],var_k)
			new_p_k=likelihood_calc.P_k(new_k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],self.Mu1_v[-1],self.Mu2_v[-1],self.timestamps,self.deltas)
			
			r=likelihood_calc.calc_ratio(new_p_k - past_p_k)
			
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
				past_p_Mu1=likelihood_calc.P_mu1(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				new_Mu1[p1]=ss.norm.rvs(self.Mu1_v[-1][p1],scale=var_mu)
				new_p_Mu1=likelihood_calc.P_mu1(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
				r=likelihood_calc.calc_ratio(new_p_Mu1-past_p_Mu1)

				if(r>=1 or ss.uniform.rvs()<r):
					pass
				else:
					new_Mu1[p1]=self.Mu1_v[-1][p1]
					self.reject_mu1[p1]+=1
				##########  ########
				
				######## Mu2 ########
				past_p_Mu2=likelihood_calc.P_mu2(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				new_Mu2[p1]=ss.norm.rvs(self.Mu2_v[-1][p1],scale=var_mu)
				new_p_Mu2=likelihood_calc.P_mu2(p1,k,self.d,self.T,self.Alpha1_v[-1],self.Alpha2_v[-1],new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
				r=likelihood_calc.calc_ratio(new_p_Mu2-past_p_Mu2)

				if(r>=1 or ss.uniform.rvs()<r):
					pass
				else:
					new_Mu2[p1]=self.Mu2_v[-1][p1]
					self.reject_mu2[p1]+=1
				##########  ########

				for p2 in range(self.d):

					##### Alpha1 #####
					past_p_Alpha1=likelihood_calc.P_alpha1(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					new_Alpha1[p2][p1]=ss.norm.rvs(self.Alpha1_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha1=likelihood_calc.P_alpha1(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					r=likelihood_calc.calc_ratio(new_p_Alpha1- past_p_Alpha1)	 
					if(r>=1 or ss.uniform.rvs()<r):
						pass
					else:
						new_Alpha1[p2][p1]=self.Alpha1_v[-1][p2][p1]
						self.reject_alpha1[p2][p1]+=1
						
					##### Alpha2 #####
					past_p_Alpha2=likelihood_calc.P_alpha2(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
					new_Alpha2[p2][p1]=ss.norm.rvs(self.Alpha2_v[-1][p2][p1],scale=var_alpha)
					new_p_Alpha2=likelihood_calc.P_alpha2(p1,p2,k,self.d,self.T,new_Alpha1,new_Alpha2,new_Mu1,new_Mu2,self.timestamps,self.deltas)
				
					r=likelihood_calc.calc_ratio(new_p_Alpha2- past_p_Alpha2)
					if(r>=1 or ss.uniform.rvs()<r):
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
		print("mu1",[i/self.n_iter for i in self.reject_mu1])
		print("mu1 average",np.mean([i/self.n_iter for i in self.reject_mu1]))
		print("mu2",[i/self.n_iter for i in self.reject_mu2])
		print("mu2 average",np.mean([i/self.n_iter for i in self.reject_mu2]))
		print("alpha1",[[i/self.n_iter for i in j] for j in self.reject_alpha1])
		print("alpha1 average",np.mean([[i/self.n_iter for i in j] for j in self.reject_alpha1]))
		print("alpha2",[[i/self.n_iter for i in j] for j in self.reject_alpha2])
		print("alpha2 average",np.mean([[i/self.n_iter for i in j] for j in self.reject_alpha2]))

	def plot_values(self):		
		plt.plot(range(len(self.k_v)),self.k_v)
		plt.title('k')
		plt.show()

		Mu1_lista=[[] for i in range(self.d)]
		Mu2_lista=[[] for i in range(self.d)]
		for i in range(self.d):
			for i_iter in range(0,self.n_iter-self.burn_in):
				Mu1_lista[i].append(self.Mu1_v[i_iter][i])
				Mu2_lista[i].append(self.Mu2_v[i_iter][i])
		for i in range(self.d):
			plt.plot(range(len(Mu1_lista[i])),Mu1_lista[i],label=str(i))
		plt.title('Mu1')
		plt.legend()
		plt.show()  

		for i in range(self.d):
			plt.plot(range(len(Mu2_lista[i])),Mu2_lista[i],label=str(i))
		plt.title('Mu2')
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
				plt.plot(range(len(Alpha1_lista[i][j])),Alpha1_lista[j][i],label='%d %d' % (j,i))
			
			plt.title('Alpha1')	
			plt.legend()
			plt.show()  
			
		for i in range(self.d):
			for j in range(self.d):
				plt.plot(range(len(Alpha2_lista[i][j])),Alpha2_lista[j][i],label='%d %d' % (j,i))
			
			plt.title('Alpha1')	
			plt.legend()
			plt.show() 
