from base.base_data_loader import BaseDataLoader

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew,kurtosis
from scipy.stats import kstest,ks_2samp
from scipy.stats import moment

from scipy.stats import entropy as kullback_leibler

def calculate_indicators(df_raw):
    array = df_raw.values
    
    mean = np.apply_along_axis(lambda x : np.mean(x[~np.isnan(x)]),1,array)
    std = np.apply_along_axis(lambda x : np.std(x[~np.isnan(x)]),1,array)
    cv = np.divide(std,mean,out=np.zeros_like(std),where=mean!=0)
    skewness = np.apply_along_axis(lambda x : skew(x[~np.isnan(x)]),1,array)
    kurtosis_coeff = np.apply_along_axis(lambda x : kurtosis(x[~np.isnan(x)]),1,array)
    max_week = np.apply_along_axis(lambda x : np.max(x[~np.isnan(x)]),1,array)
    ratio_max_mean = np.divide(max_week,mean,out=np.zeros_like(max_week),where=mean!=0)
    df = pd.DataFrame({'mean':mean,'cv':cv,
                        'skewness':skewness,
                        'kurtosis':kurtosis_coeff,
                        'max_week':max_week,
                        'ratio_max_mean':ratio_max_mean})
    df.index = df_raw.index
    return df

def indicator_distance(ind_r, ind_f, dist="kl"):
    ind_r = np.arcsinh(ind_r)
    ind_f = np.arcsinh(ind_f)
    
    _, bins = np.histogram(np.concatenate([ind_r, ind_f]), bins =50)
    
    vals_r, _ = np.histogram(ind_r , bins = bins)
    vals_f, _ = np.histogram(ind_f , bins = bins)

    if dist == "kl":
        vals_r = vals_r + 0.1
        vals_f = vals_f + 0.1
    vals_r = vals_r/vals_r.sum()
    vals_f = vals_f/vals_f.sum()
    
    if dist == "kl":
        return kullback_leibler(vals_r,vals_f)
    else:
        return -np.log(np.sqrt(vals_f*vals_r).sum())

class SmartmeterLoader(BaseDataLoader):
    def __init__(self, config):
        super(SmartmeterLoader, self).__init__(config)
        
        self.log.info('loading data ...')
        # processed data 
        raw_data =   pd.read_csv(self.config.data['file_data'], index_col=[0,1])
        
        # data indicators and labels
        indicators = pd.read_csv(self.config.data['file_data_labels'], index_col=[0,1])
        
        self.log.info('dataset size %d'% (raw_data.shape[0],))

        # filtering by specified labels
        'stdorToU','Acorn', 'Acorn_grouped'
        
        if 'label_classes_stdorToU' in self.config.data:
            indicators = indicators[indicators['stdorToU'].isin(self.config.data['label_classes_stdorToU'])]    
        if 'label_classes_Acorn' in self.config.data:
            indicators = indicators[indicators['Acorn'].isin(self.config.data['label_classes_Acorn'])]
        if 'label_classes_Acorn_grouped' in self.config.data:
            indicators = indicators[indicators['Acorn_grouped'].isin(self.config.data['label_classes_Acorn_grouped'])]
        
        # stdorToU Acorn Acorn_grouped
        
        if 'label_classes_week' in self.config.data :
            if self.config.data['label_classes_cond_week'] == "all":
                indicators = indicators[indicators.index.get_level_values('week').isin(self.config.data['label_classes_week'])]
            elif self.config.data['label_classes_cond_week'] == "not":
                indicators = indicators[~indicators.index.get_level_values('week').isin(self.config.data['label_classes_week'])]
            elif self.config.data['label_classes_cond_week'] == "lower":
                indicators = indicators[indicators.index.get_level_values('week')<=self.config.data['label_classes_week']]
            elif self.config.data['label_classes_cond_week'] == "higher":
                indicators = indicators[indicators.index.get_level_values('week')>self.config.data['label_classes_week']]
        
        if 'ignore_null' in self.config.data:
            indicators = indicators.loc[indicators['mean']>0]
        
        raw_data   = raw_data.loc[indicators.index]
        
        self.indicators = indicators
        self.raw_data   = raw_data

        self.log.info('filtered dataset size %d'% (raw_data.shape[0],))
        
        #if 'PS' in self.config.data['label']:
        #    assert indicators['PS'].unique().shape[0] == len(self.config.data['label_classes_ps']) , 'Error : no occurences for some labels PS'
        #if 'PROFIL_RFX' in self.config.data['label']:
        #    assert indicators['PROFIL_RFX'].unique().shape[0] == len(self.config.data['label_classes_res']) , 'Error : no occurences for some labels PROFIL_RFX'
        
        data = self.raw_data.values
        
        self.min = data.min()
        self.max = data.max()
        self.log.info('transforming data')
        
        data_classes = None
        if 'label' in self.config.data:
            discrete_classes = list(set(['stdorToU','Acorn', 'Acorn_grouped']) & set(self.config.data['label'])) 
            if len(discrete_classes)!=0:
                self.classes = self.indicators[discrete_classes].drop_duplicates().values
                self.counts = [self.indicators[discrete_classes].isin(c).all(1).sum() for c in self.classes]
                
                self.labels = []
                sub_classes = [self.indicators[lab].drop_duplicates().values.tolist() for lab in discrete_classes]
                
                sub_labels = [np.eye(len(sub_c)) for sub_c in sub_classes]
                for c in self.classes:
                    sub_indexes = [sub_c.index(s_c) for s_c,sub_c in zip(c, sub_classes)]
                    self.labels.append(np.concatenate([sub_l[sub_i] for sub_i, sub_l in zip(sub_indexes,sub_labels)]))
                    
                self.labels = np.array(self.labels)
                #self.labels = np.eye(len(self.classes))
                
                self.gen_probabilities = np.array(self.counts)/raw_data.shape[0]
                data_classes = indicators[discrete_classes].values

            if 'week' in self.config.data['label']:
                self.week_classes = np.arange(1,53)
                self.counts = self.indicators.index.get_level_values('week').value_counts().sort_index().values
                self.week_gen_probabilities = np.array(self.counts)/raw_data.shape[0]
                self.week_labels = np.array(list(zip(np.sin(2*np.pi*self.week_classes/52), np.cos(2*np.pi*self.week_classes/52))))
                if data_classes is None:
                    data_classes = indicators.index.get_level_values('week').values
                else:
                    data_classes = np.concatenate([data_classes, np.expand_dims(indicators.index.get_level_values('week').values,axis=1)], axis=1)


        trans_out = self.transform(data, data_classes)

        data = trans_out['data']
        data_labels = trans_out['labels']
        #data = np.hstack((data, np.tile(data[:, [-1]], 3)))

        self.log.info('shuffling data')
          
        s = np.arange(data.shape[0])
        
        seed = 123
        np.random.seed(seed)
        np.random.shuffle(s)
        np.random.seed()
        
        data = data[s]

        self.test_size = int(self.config.data['test_ratio']*data.shape[0])
        self.train_size = data.shape[0] - self.test_size

        self.train_data = data[:self.train_size]
        self.test_data = data[-self.test_size:]
                
        if 'label' in self.config.data:
            data_labels = data_labels[s]
            self.train_data_labels = data_labels[:self.train_size]
            self.test_data_labels = data_labels[-self.test_size:]

        self.train_counter = 0
        self.test_counter = 0

            
    def next_train_batch(self, batch_size):
        if self.train_counter + batch_size > self.train_data.shape[0]:
            self.train_counter = 0 
        from_c = self.train_counter  
        to_c = self.train_counter + batch_size
        self.train_counter = self.train_counter + batch_size

        out_data = self.train_data[from_c:to_c]
        out_labels = self.train_data_labels[from_c:to_c] if('label' in self.config.data) else None

        yield {'data': out_data,'labels':out_labels}

    def next_test_batch(self, batch_size):
        if self.test_counter + batch_size > self.test_data.shape[0]:
            self.test_counter = 0
        
        from_c = self.test_counter  
        to_c = self.test_counter + batch_size
        self.test_counter = self.test_counter + batch_size

        out_data = self.test_data[from_c:to_c]
        out_labels = self.test_data_labels[from_c:to_c] if('label' in self.config.data) else None

        yield {'data': out_data,'labels':out_labels}

    def sample_random_labels(self, batch_size, uniform_probs=False):
        """ sample random labels from the dataset
        Args:
            batch_size : number of generated labels
            uniform_probs : false for sampling labels from the dataset distribution 
        """
        labels = None
        discrete_classes = list(set(['stdorToU','Acorn', 'Acorn_grouped']) & set(self.config.data['label'])) 
        
        if uniform_probs:
            if 'week' in self.config.data['label']:
                labels = self.week_labels[np.random.choice(len(self.labels), batch_size)]
            if len(discrete_classes)>0:
                if labels is None:
                    labels = self.labels[np.random.choice(len(self.labels), batch_size)]
                else:
                    labels = np.concatenate([self.labels[np.random.choice(len(self.labels), batch_size)],labels],axis=1)    
            
        else:
            if 'week' in self.config.data['label']:
                labels = self.week_labels[np.random.choice(len(self.labels), batch_size, p=self.week_gen_probabilities)]
            if len(discrete_classes)>0:
                if labels is None:
                    labels = self.labels[np.random.choice(len(self.labels), batch_size, p=self.gen_probabilities)]
                else:
                    labels = np.concatenate([self.labels[np.random.choice(len(self.labels), batch_size, p=self.gen_probabilities)],labels],axis=1)    
            
        return labels 


    def transform(self, data, classes=None, phcs=None):
        """ Transform dataset curves and labels to neural network input
        Args:
            data : np array of curves
            labels : np array of labels 
        """
        out_labels = None if classes is None else self.transform_labels(classes)
        return {'data': self.transform_data(data),'labels':out_labels}

    def transform_data(self, data_):
        data = data_
        if self.config.data['transform'] == 'boxcox':
            min_scaled = boxcox(self.min+0.01, self.config.data['transform_lambda'])
            max_scaled = boxcox(self.max+0.01, self.config.data['transform_lambda'])
            data = boxcox(data+0.01, self.config.data['transform_lambda'])
        elif self.config.data['transform'] == 'ihs':
            min_scaled = np.arcsinh(self.min*self.config.data['transform_lambda'])/self.config.data['transform_lambda']
            max_scaled = np.arcsinh(self.max*self.config.data['transform_lambda'])/self.config.data['transform_lambda']
            data = np.arcsinh(data*self.config.data['transform_lambda'])/self.config.data['transform_lambda']
        else:
            min_scaled = self.min
            max_scaled = self.max

        if self.config.data['normalize']:
            data = 2*(data - min_scaled)/(max_scaled - min_scaled)-1

        return data

    def transform_labels(self, classes):
        labels = None
        discrete_classes = list(set(['stdorToU','Acorn', 'Acorn_grouped']) & set(self.config.data['label'])) 
        if 'week' in self.config.data['label']:
            if len(classes.shape) < 2:
                w_classes = np.array(classes)-1
            else:
                w_classes = np.array(classes[:,-1])-1
                l_classes = classes[:,:-1]
            labels = self.week_labels[w_classes.tolist()]


        if len(discrete_classes)>0:
            if labels is None:
                labels = self.labels[list(map(self.classes.tolist().index,classes.tolist()))]               
            else: 
                labels = np.concatenate([self.labels[list(map(self.classes.tolist().index,l_classes.tolist()))], labels], axis=1)
        
        return labels
        
    def inverse_transform(self, data, labels=None, masks=None):
        """ Transform neural network output to regular data
        Args:
            data : np array of curves
            labels : np array of labels 
        """
        out_labels = None if labels is None else self.inverse_transform_labels(labels)
        return {'data': self.inverse_transform_data(data),'labels':out_labels}

    def inverse_transform_data(self, data_):
        data = data_
        if self.config.data['transform'] == 'ihs':
            min_scaled = np.arcsinh(self.min*self.config.data['transform_lambda'])/self.config.data['transform_lambda']
            max_scaled = np.arcsinh(self.max*self.config.data['transform_lambda'])/self.config.data['transform_lambda']

            if self.config.data['normalize']:
                data = ((data +1)/2)*(max_scaled-min_scaled)+min_scaled

            data = np.sinh(data*self.config.data['transform_lambda'])/self.config.data['transform_lambda']
        
        elif self.config.data['transform'] == 'boxcox':
            min_scaled = boxcox(self.min+0.01, self.config.data['transform_lambda'])
            max_scaled = boxcox(self.max+0.01, self.config.data['transform_lambda'])

            if self.config.data['normalize']:
                data = ((data +1)/2)*(max_scaled-min_scaled)+min_scaled

            data = inv_boxcox(data+0.01, self.config.data['transform_lambda'])

        return data

    def inverse_transform_labels(self, labels):
        classes = None
        discrete_classes = list(set(['stdorToU','Acorn', 'Acorn_grouped']) & set(self.config.data['label'])) 
        
        if 'week' in self.config.data['label']:
            w_labels = labels[:,-2:]
            l_labels = labels[:,:-2]
            classes = list(map(lambda x:self.week_labels.tolist().index(x)+1,w_labels.tolist()))
            

        if len(discrete_classes) >0:
            
            if classes is None:                
                classes = self.classes[list(map(self.labels.tolist().index,labels.tolist()))]
                         
            else: 
                classes = np.concatenate([self.classes[list(map(self.labels.tolist().index,l_labels.tolist()))], np.expand_dims(classes, axis=1)], axis=1)
                
        return classes


    def test_similarity(self, data_):
        """ Test similarity of generated curves to the dataset
        Args:
            data_: dict with keys data containing curves and labels containing classes
        """
        if 'label' in self.config.data:
            fake_data = data_['data']
            fake_labels = data_['labels']
            fake_inds = calculate_indicators(pd.DataFrame(fake_data))
            fake_labels = pd.DataFrame(fake_labels)
            fake_labels.columns = self.config.data['label']
            
            fake_inds = pd.concat([fake_inds,fake_labels], axis=1)
            
            tests = pd.DataFrame(columns = ['mean','cv','skewness', 'kurtosis','max_week','ratio_max_mean'])
            i = 0
            for c in self.classes:
                indicators_tests = []
                #for ind in tests.columns:
                #    _, indicators_test = ks_2samp(self.indicators.loc[self.indicators[self.config.data['label']].isin(c).all(1)][ind],\
                #                                        fake_inds.loc[   fake_inds[self.config.data['label']].isin(c).all(1)][ind])
                #    indicators_tests.append(indicators_test)
                
                #mean_inds_real = self.indicators.loc[self.indicators[self.config.data['label']].isin(c).all(1)][tests.columns].mean(axis=0).values
                #mean_inds_fake = fake_inds[self.config.data['label']].isin(c).all(1)][tests.columns].mean(axis=0).values

                inds_real = self.indicators.loc[self.indicators[self.config.data['label']].isin(c).all(1)][tests.columns]
                inds_fake = fake_inds.loc[fake_inds[self.config.data['label']].isin(c).all(1)] [tests.columns]
                tests.loc[i] = [indicator_distance(inds_real[ind], inds_fake[ind], dist="kl") for ind in tests.columns]
                
                i+=1
            return {"tests":tests, "indicators": fake_inds}
        else:
            fake_data = data_['data']
            inds_fake = calculate_indicators(pd.DataFrame(fake_data))
            inds_real = self.indicators[inds_fake.columns]
            indicators_tests = pd.DataFrame([[indicator_distance(inds_real[ind], inds_fake[ind], dist="kl") for ind in inds_fake.columns]],
                                            columns = inds_fake.columns)
            
            return {"tests":indicators_tests, "indicators": inds_fake}





