import numpy as np
from sklearn.preprocessing import PowerTransformer
from matplotlib.ticker import ScalarFormatter


# ---- # ---- # ---- # OutliersFixer # ---- # ---- # ---- #

class OutliersFixer:
    
    ''' 
       Objects of this class get as input 2D float arrays (n_samples x n_features).
       Columns shall first be made Gaussian-like (e.g. by use of the PowerTransform),
       as otherwise extreme quantiles might have different interpretations...
    '''
    
    def __init__(self, method='zscore'):
        self.method_ = method
        if method=='zscore':
            self.mean_ = []
            self.std_ = []
        elif method=='quantiles':
            self.ql_val_ = []
            self.qh_val_ = []
        else:
            raise ValueError('Class method not understood. Available methods: \'quantiles\' or \'zscore\' (default).')
        
        
    def fit(self, data, q_low, q_high):
        
        if len(data.shape)!=2:
            raise ValueError('Inserted data is not 2D (n_samples x n_features)')
            
        if self.method_=='quantiles':
            self.ql_val_ = np.quantile(data, q_low, axis=0)
            self.qh_val_ = np.quantile(data, q_high, axis=0)

            self.ql_val_prefl_ = []
            self.ql_val_prefh_ = []
            self.qh_val_prefl_ = []
            self.qh_val_prefh_ = []

            for element in self.ql_val_:
                if element<0:
                    self.ql_val_prefl_.append(1.2)
                    self.ql_val_prefh_.append(1.)
                else:
                    self.ql_val_prefl_.append(1.)
                    self.ql_val_prefh_.append(1.2)


            for element in self.qh_val_:
                if element<0:
                    self.qh_val_prefl_.append(1.2)
                    self.qh_val_prefh_.append(1.)
                else:
                    self.qh_val_prefl_.append(1.)
                    self.qh_val_prefh_.append(1.2)
        
        elif self.method_=='zscore':
            self.mean_ = np.nanmean(data, axis=0)
            self.std_ = np.nanstd(data, axis=0)
            self.ql_val_ = q_low
            self.qh_val_ = q_high
            
    def transform(self, data, seed=0):
        
        if len(data.shape)!=2:
            raise ValueError('Inserted data is not 2D (n_samples x n_features)')
        
        if self.method_=='quantiles':
            if (len(self.ql_val_)>0) & (len(self.qh_val_)>0):
                data_fixed = data.copy()
                #np.random.seed(seed)
                for i in range(len(self.ql_val_)):
                    ql_val = self.ql_val_[i]
                    taill_size = sum(data_fixed[:, i]<ql_val)
                    if taill_size>0:                    
                        taill_l, taill_h = self.ql_val_prefl_[i]*ql_val, self.ql_val_prefh_[i]*ql_val
                        lower_tail = np.random.uniform(taill_l, taill_h, taill_size)
                        data_fixed[data_fixed[:, i]<ql_val, i] = lower_tail

                    qh_val = self.qh_val_[i]
                    tailh_size = sum(data_fixed[:, i]>qh_val)
                    if tailh_size>0:
                        tailh_l, tailh_h = self.qh_val_prefl_[i]*qh_val, self.qh_val_prefh_[i]*qh_val
                        upper_tail = np.random.uniform(tailh_l, tailh_h, tailh_size)
                        data_fixed[data_fixed[:, i]>qh_val, i] = upper_tail

                return data_fixed
            else:
                raise ValueError('OutliersFixer not fitted')
                
        elif self.method_=='zscore':
            if (len(self.mean_)>0) & (len(self.std_)>0):
                data_fixed = data.copy()
                pt = PowerTransformer()
                data_normal = pt.fit_transform(data.copy())
                #np.random.seed(seed)
                z_values = data_normal
                for i in range(data.shape[1]):
                    tail_size = sum(z_values[:, i]<self.ql_val_)
                    if tail_size>0:
                        min_val = self.std_[i]*np.sign(self.ql_val_)*(abs(self.ql_val_)+0.5) + self.mean_[i]
                        max_val = self.std_[i]*self.ql_val_ + self.mean_[i]
                        lower_tail = np.random.uniform(min_val, 
                                                       max_val,
                                                       tail_size)
                        data_fixed[z_values[:, i]<self.ql_val_, i] = lower_tail
                    
                    
                    tail_size = sum(z_values[:, i]>self.qh_val_)
                    if tail_size>0:
                        min_val = self.std_[i]*self.qh_val_ + self.mean_[i]
                        max_val = self.std_[i]*np.sign(self.qh_val_)*(abs(self.qh_val_)+0.5) + self.mean_[i]
                        upper_tail = np.random.uniform(min_val,
                                                       max_val, 
                                                       tail_size)
                        data_fixed[z_values[:, i]>self.qh_val_, i] = upper_tail
                    
                return data_fixed
            else:
                raise ValueError('OutliersFixer not fitted')
            
    
    def fit_transform(self, data, q_low, q_high, seed=0):
        
        if len(data.shape)!=2:
            raise ValueError('Inserted data is not 2D (n_samples x n_features)')
            
        if self.method_=='quantiles':
            ## fit ##
            self.ql_val_ = np.quantile(data, q_low, axis=0)
            self.qh_val_ = np.quantile(data, q_high, axis=0)

            self.ql_val_prefl_ = []
            self.ql_val_prefh_ = []
            self.qh_val_prefl_ = []
            self.qh_val_prefh_ = []

            for element in self.ql_val_:
                if element<0:
                    self.ql_val_prefl_.append(1.2)
                    self.ql_val_prefh_.append(1.)
                else:
                    self.ql_val_prefl_.append(1.)
                    self.ql_val_prefh_.append(1.2)


            for element in self.qh_val_:
                if element<0:
                    self.qh_val_prefl_.append(1.2)
                    self.qh_val_prefh_.append(1.)
                else:
                    self.qh_val_prefl_.append(1.)
                    self.qh_val_prefh_.append(1.2)


            ## transform ##   
            data_fixed = data.copy()
            #np.random.seed(seed)
            for i in range(len(self.ql_val_)):
                ql_val = self.ql_val_[i]
                taill_size = sum(data_fixed[:, i]<ql_val)
                if taill_size>0:                    
                    taill_l, taill_h = self.ql_val_prefl_[i]*ql_val, self.ql_val_prefh_[i]*ql_val
                    lower_tail = np.random.uniform(taill_l, taill_h, taill_size)
                    data_fixed[data_fixed[:, i]<ql_val, i] = lower_tail

                qh_val = self.qh_val_[i]
                tailh_size = sum(data_fixed[:, i]>qh_val)
                if tailh_size>0:
                    tailh_l, tailh_h = self.qh_val_prefl_[i]*qh_val, self.qh_val_prefh_[i]*qh_val
                    upper_tail = np.random.uniform(tailh_l, tailh_h, tailh_size)
                    data_fixed[data_fixed[:, i]>qh_val, i] = upper_tail

            return data_fixed
        
        elif self.method_=='zscore':
            ## fit ##
            self.mean_ = np.nanmean(data, axis=0)
            self.std_ = np.nanstd(data, axis=0)
            self.ql_val_ = q_low
            self.qh_val_ = q_high


            ## transform ##   
            data_fixed = data.copy()
            pt = PowerTransformer()
            data_normal = pt.fit_transform(data.copy())
            #np.random.seed(seed)
            z_values = data_normal
            for i in range(data.shape[1]):
                tail_size = sum(z_values[:, i]<self.ql_val_)
                if tail_size>0:
                    min_val = self.std_[i]*np.sign(self.ql_val_)*(abs(self.ql_val_)+0.5) + self.mean_[i]
                    max_val = self.std_[i]*self.ql_val_ + self.mean_[i]
                    lower_tail = np.random.uniform(min_val, 
                                                   max_val,
                                                   tail_size)
                    data_fixed[z_values[:, i]<self.ql_val_, i] = lower_tail


                tail_size = sum(z_values[:, i]>self.qh_val_)
                if tail_size>0:
                    min_val = self.std_[i]*self.qh_val_ + self.mean_[i]
                    max_val = self.std_[i]*np.sign(self.qh_val_)*(abs(self.qh_val_)+0.5) + self.mean_[i]
                    upper_tail = np.random.uniform(min_val,
                                                   max_val, 
                                                   tail_size)
                    data_fixed[z_values[:, i]>self.qh_val_, i] = upper_tail

            return data_fixed


# ---- # ---- # ---- # ScalarFormatterClass # ---- # ---- # ---- #


class ScalarFormatterClass(ScalarFormatter):
    def __init__(self, n=1, m=1):
        self.n_ = n
        self.m_ = m
        super(ScalarFormatterClass, self).__init__()
    def _set_format(self, n=None, m=None):
        if n is None:
            n=self.n_
        if m is None:
            m=self.m_
        s = '%d.%d' % (n, m)
        self.format = '%' + s + 'f'