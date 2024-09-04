import pandas as pd
import numpy as np
import re
import pickle
import ast


class PredictionProcessor():
    def __init__(self, ouput_path, pkl_path, type):
        #self.ref_dict=ref_dict
        '''
        translate is the kmers of the sequence
        prediction is the predicted label of the ml
        predictions are the true labels
        seq is the sequence of the transcript        
        '''
        self.type=type
        self.df = pd.read_csv(ouput_path)
        self.df.drop(columns=['input_ids','token','labels'], inplace=True)      
        #self.df=self.df[self.df['t_name']=='ENST00000299381']
        self.df['translate_input'] = self.df['translate_input'].apply(lambda x: x.split(' '))
        self.df['true_labels'] = self.df['true_labels'].apply(ast.literal_eval)
        self.df['predictions'] = self.df['predictions'].apply(ast.literal_eval)
        self.df.set_index('t_name', inplace=True)
        self.input_data=self.df.to_dict(orient='index')


        self.START=['ATG','GTG','TTG','CTG']
        self.STOP=['TAA','TAG','TGA']
        self.pattern = re.compile('|'.join(self.START))
        # print(self.input_data)
        self.output={}
        del self.df
        with open(pkl_path, 'rb') as file:
            self.ref_dict=pickle.load(file)
    def retrieve_orf_data(self):
        if self.type=='cov':
            for transcript in self.input_data:
                # print(transcript)
                self.t_tis={}
                self.weird={}
                self.stops=[]
                self.unclear_tis={}
                self.transcript=transcript
                true_labels=self.input_data[transcript]['true_labels']
                translate=self.input_data[transcript]['translate_input']
                for idx, value in enumerate(true_labels[:-1]):
                    self.idx=idx
                    if value!=0:
                        if idx==0:
                            self.find_tis_in_kmer(translate[idx]+translate[idx+1], value)
                        if value > true_labels[idx-1]:
                            self.find_tis_in_kmer(translate[idx]+translate[idx+1], value-true_labels[idx-1])
                        if value > true_labels[idx+1]:
                            for _ in range(value-true_labels[idx+1]):
                                self.stops.append(idx+1)
                if true_labels[-1] >0:
                    for _ in range(true_labels[-1]):
                        self.stops.append(len(true_labels))
                self.sort_unclear_tis()
        elif self.type=='tis':
                for transcript in self.input_data:
                # print(transcript)
                    self.t_tis={}
                    self.stops=[]
                    self.unclear_tis={}
                    self.transcript=transcript
                    true_labels=self.input_data[transcript]['true_labels']
                    translate=self.input_data[transcript]['translate_input']
                    for idx, value in enumerate(true_labels[:-1]):
                        self.idx=idx
                        if value!=0:
                            # print(value, idx, translate[idx])
                            self.find_tis_in_kmer(translate[idx]+translate[idx+1], value)


    def find_tis_in_kmer(self, kmer, n_tis):
        back=kmer
        self.tis={}
        kmer = kmer[:8]
        n=0
        last=0
        while self.pattern.search(kmer) is not None:
            match = self.pattern.search(kmer)
            if last!=match.start()+1+(self.idx*6)-3: #this ignore a tis if there a tis in the same frame before in the same kmer
                last=match.start()+1+(self.idx*6)
                self.tis[self.transcript+'_'+str(self.idx)+'_'+str(n)]={'start':match.start()+1+(self.idx*6), 'transcript':self.transcript}
                kmer = ('N' * (match.start()+1)) + kmer[match.start()+1:]
                n+=1
            else:
                break
        self.find_tts()
        if self.type=='cov':
            if len(self.tis) == n_tis:
                self.output.update(self.tis)
                self.t_tis.update(self.tis)
            else:
                self.unclear_tis.update(self.tis)
        elif self.type=='tis':
            for i in self.tis:
                if 'stop' in self.tis[i].keys():
                    self.output[i]=self.tis[i]

    def sort_unclear_tis(self):
        for orf_id in self.t_tis.keys():
            self.stops.remove(self.output[orf_id]['stop_idx']) 
        if len(self.unclear_tis)>0:
            for orf_id in self.unclear_tis.keys():
                if 'stop_idx' in self.unclear_tis[orf_id].keys():
                    if self.unclear_tis[orf_id]['stop_idx'] in self.stops:
                        self.output[orf_id]=self.unclear_tis[orf_id]
                        self.stops.remove(self.unclear_tis[orf_id]['stop_idx'])
                    else:
                        self.weird[orf_id]=self.unclear_tis[orf_id]

    def find_tts(self):
        seq=self.input_data[self.transcript]['sequence']
        for orf_id in self.tis.keys():
            self.orf_id=orf_id
            start=self.tis[orf_id]['start']
            for nu in range(start-1, len(seq), 3):
                codon=seq[nu:nu+3]
                if codon in self.STOP and self.type=='cov':
                    self.check_tts_label_down(nu) #here we give the index not the stop position
                    if self.correct==True:
                        break
                    else:
                        continue
                elif codon in self.STOP and self.type=='tis':
                    self.tis[orf_id]['stop']=nu+1
                    break
                

    def check_tts_label_down(self,stop ):
        '''Here we are looking for TTS that have a label level down
        First we make sure that the stop isnt the last kmer of the sequence
        we use +2 to get the last nu of the stop codon as to make sure to get the right kmer in case the stop overlap 2 kmer
        then we check if the kmer +1 has a label down
        
        '''
        self.correct=False
        start=['ATG','GTG','TTG','CTG']
        translate=self.input_data[self.transcript]['translate_input']
        predictions=self.input_data[self.transcript]['true_labels']
        lenght=[]
        """
        First we check if we are in a situation where single token can happen (ie end of sequence)
        if tis the case qwe have an algo that find the tts 
        otherwise we find the right tts
        """
        lenght=len(predictions)
        if (stop/6)>(lenght-5):
            for idx,i in enumerate(predictions[::-1]):
                if i>0:
                    label_down=lenght-1-idx
                    break
                elif predictions[lenght-2-idx]>i:
                    label_down=lenght-2-idx
                    break
                elif idx>10: #break if we dont find a label down in the 10 last kmers
                    # print('no label down found')
                    return
            if len(translate[label_down])==1:
                for jdx,j in enumerate(translate[label_down::-1]):
                    if len(j)==6:
                        last_six_mer=label_down-jdx
                        s=(last_six_mer+1)*6+jdx-3
                        if stop==s:
                            self.tis[self.orf_id]['stop']=stop+1
                            self.tis[self.orf_id]['stop_idx']=label_down+1
                            self.correct=True
                        break        
            else:
                self.tis[self.orf_id]['stop']=stop+1
                self.tis[self.orf_id]['stop_idx']=int(((stop)+2)/6)+1
        elif predictions[int((stop+2)/6)] > predictions[int((stop+2)/6)+1]:
            self.tis[self.orf_id]['stop']=stop
            self.tis[self.orf_id]['stop_idx']=int(((stop)+2)/6)+1
            self.correct=True

    def get_stats(self):
        stats={}
        stats['ref_%']=self.compare_ref()
        stats['label_balance']=self.get_label_balance()
        stats['orf_type']=self.find_class()
        stats['n_prediction']=len(self.output)
        print(self.output) 
        return stats
        
    def compare_ref(self):
        self.ref_set=set([(i,self.ref_dict[i]['start']) for i in self.ref_dict.keys()])
        output_set=[(i['transcript'],i['start']) for i in self.output.values()]
        output_set=set(output_set)
        intersection=self.ref_set.intersection(output_set)
        print(len(intersection)/len(self.ref_set)*100)
        ref_retrieved_p100=len(intersection)/len(self.ref_set)*100
        return ref_retrieved_p100
    def find_class(self):
        for i in self.output:
            transcript=self.output[i]['transcript']
            #Check if orf is ref
            if (self.output[i]['transcript'], self.output[i]['start']) in self.ref_set:
                self.output[i]['class']='Annotated CDS'
            #TODO logic for orf class
            #upstream or downstream
            elif transcript not in self.ref_dict.keys():
                self.output[i]['class']='lncrna'
            elif self.output[i]['start']<self.ref_dict[transcript]['start']:
                if self.output[i]['stop']<self.ref_dict[transcript]['start']:
                    self.output[i]['class']='uorf'
                else:
                    if self.output[i]['start']%3==self.ref_dict[transcript]['start']%3:
                        self.output[i]['class']='n-terminal extension'
                    else:
                        self.output[i]['class']='uoorf'
            elif self.output[i]['start']>self.ref_dict[transcript]['start']:
                if self.output[i]['start']<self.ref_dict[transcript]['stop']:
                    if self.output[i]['start']%3==self.ref_dict[transcript]['start']%3:
                        self.output[i]['class']='n-terminal truncation'
                    else:
                        self.output[i]['class']='doorf'
                else:
                    self.output[i]['class']='dorf'
        class_counts = {
            'Annotated CDS': 0,
            'lncrna': 0,
            'uorf': 0,
            'n-terminal extension': 0,
            'uoorf': 0,
            'n-terminal truncation': 0,
            'doorf': 0,
            'dorf': 0
        }
        for _, value in self.output.items():
            class_type = value['class']
            if class_type in class_counts:
                class_counts[class_type] += 1
        return class_counts
    def get_label_balance(self):
        balance=[]
        for i in self.input_data:
            if sum(self.input_data[i]['predictions']) !=0:
                bb=self.input_data[i]['predictions'].count(1)
                cc=len(self.input_data[i]['predictions'])

                balance.append(bb/cc*100)
        return np.mean(balance)
        
        


# a=PredictionProcessor()
# a.retrieve_orf_data()
# a.compare_ref()
# a.find_class()
# a.get_label_balance()