# -*- coding: utf8

from bisect import bisect
import numpy as np
cimport numpy as np


cdef total_intensity2(float t, np.ndarray Alpha_ba, np.ndarray Beta_ba, np.ndarray mu_rates, past):
    cdef np.ndarray lambdas_t = np.zeros(mu_rates.shape[0], dtype=np.float)
    cdef float tp, tpp, busca_rate
    cdef int tpp_idx
    cdef int n_nodes= mu_rates.shape[0]
    
    for proc_a in range(n_nodes):
        lambdas_t[proc_a] = mu_rates[proc_a]
        if len(past[proc_a]) == 0:
            continue

        tp = past[proc_a][-1]
        assert tp <= t
        for proc_b in range(n_nodes):
            if len(past[proc_b]) == 0:
                continue

            tpp_idx = bisect(past[proc_b], tp)
            if tpp_idx == len(past[proc_b]):
                tpp_idx -= 1
            tpp = past[proc_b][tpp_idx]
            while tpp >= tp and tpp_idx > 0:
                tpp_idx -= 1
                tpp = past[proc_b][tpp_idx]
            if tpp >= tp:
                continue
            busca_rate = Alpha_ba[proc_b, proc_a]
            busca_rate /= (Beta_ba[proc_b, proc_a] + tp - tpp)
            lambdas_t[proc_a] += busca_rate
    return lambdas_t 

class GrangeBuscaSimulator(object):

    def __init__(self, mu_rates, Alpha_ba, Beta_ba=None, thinning=False):
        self.mu_rates = np.asanyarray(mu_rates, dtype=np.float)
        self.n_nodes = self.mu_rates.shape[0]
        self.Alpha_ba = np.asanyarray(Alpha_ba, dtype=np.float)
        if Beta_ba is not None:
            self.Beta_ba = np.asanyarray(Beta_ba)
        else:
            self.Beta_ba = np.ones(shape=self.Alpha_ba.shape)
        self.past = [[] for i in range(self.n_nodes)]
        self.integrals = [[] for i in range(self.n_nodes)]
        self.upper_bound = 0.0
        self.current_Alpha=self.Alpha_ba
        for proc_a in range(self.n_nodes):
            self.upper_bound += self.mu_rates[proc_a]
            for proc_b in range(self.n_nodes):
                self.upper_bound += self.Alpha_ba[proc_b, proc_a] / \
                    self.Beta_ba[proc_b, proc_a]
        self.thinning = thinning
        self.t = 0

    def simulate(self, forward):
        cdef float t = self.t        
        cdef float max_time = t + forward
        cdef float dt
        
        while t < max_time:
            lambdas_t = total_intensity2(t, self.current_Alpha, self.Beta_ba, self.mu_rates, self.past)
            
            self.current_Alpha= 0.995*self.current_Alpha + 0.005*np.array([np.random.dirichlet(np.ones(self.n_nodes)) for i in range(self.n_nodes)])

            sum_lambdas_t = lambdas_t.cumsum()
            if self.thinning:
                dt = np.random.exponential(1.0 / self.upper_bound)
            else:
                dt = np.random.exponential(1.0 / sum_lambdas_t[-1])

            t = t + dt
            if t > max_time:
                break

            if self.thinning:
                if np.random.rand() < (sum_lambdas_t[-1] / self.upper_bound):
                    continue

            i = 0
            u = np.random.rand() * sum_lambdas_t[-1]
            while i < self.n_nodes:
                if sum_lambdas_t[i] >= u:
                    break
                i += 1
            if len(self.past[i]) > 1:
                self.integrals[i].append(lambdas_t[i] * (t - self.past[i][-1]))
            self.past[i].append(t)
        self.t = t
        print(self.current_Alpha)
        return self.past
