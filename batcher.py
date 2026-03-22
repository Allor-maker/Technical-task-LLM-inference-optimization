import pandas as pd
import numpy as np
A = 100#image preprocessing with certain cost of A (e.g. in milliseconds)
X = 2#image preprocessing with certain memory footprint of X
B = 100#context token has associated cost of B
Y = 2#context token has associated footprint of Y
C = 100# generated token costs C
Z = 2#generated token footprint of Z


####------Reading-Data-------
class Request:
    def __init__(self, line):
        self.status = 'start'
        self.timestep = line['TIMESTAMP']
        self.num_images = line['NumImages']
        self.num_context_tokens = line['ContextTokens']
        self.num_gen_tokens = line['GeneratedTokens']
        self.offset_ms = line['offset_ms']
        self.end_stage_2 = 0 #offset of finisghing stage 2
        self.TTFT = 0
        self.T = 0
        self.ticks = {
            'stage1':  A * self.num_images, #time preproc in ms
            'stage2' : B * self.num_context_tokens, #time prefil in ms
            'stage3' : C * self.num_gen_tokens #time generate in ms
        }
        self.costs = {
            'stage1' : 0,
            'stage2' : 0,
            'stage3' : 0,
        }

        self.footprint = self.num_images * X + self.num_context_tokens * Y + self.num_gen_tokens*Z
    
    def calc_costs(self):

        self.costs = {
            'stage1' : 0 if self.ticks['stage1'] == 0 else self.num_images / self.ticks['stage1'],
            'stage2' : 0 if self.ticks['stage2'] == 0 else self.num_context_tokens / self.ticks['stage2'],
            'stage3' : 0 if self.ticks['stage3'] == 0 else self.num_gen_tokens / self.ticks['stage3'],
        }

    def set_status(self, status):
        self.status = status
    
    def calc_ttft(self):
        self.TTFT = self.end_stage_2 - self.offset_ms

    def calc_t(self):
        self.T = 0 if self.num_gen_tokens == 0 else self.ticks['stage3'] / self.num_gen_tokens

    def get_footprint(self):
        return self.footprint
    
    def get_offset(self):
        return self.offset_ms
    
    def print_req(self):
        print(f'Request with timestep {self.timestep} has {self.num_images} images, {self.num_context_tokens} text tokens and {self.num_gen_tokens} gen tokens')
#df = pd.read_csv("AzureLMMInferenceTrace_multimodal.csv", sep=';')

#requests = np.array([Request(line) for _, line in df[1:].iterrows()])
#n = requests.size

N = 10#num accelerators
M = 350000#available memory
K = 1000#computing capability

class Batch:
    def __init__(self):
        self.curr_footprint = 0
        self.batch = np.array([])
        self.status = 0

    def create_agregation(self):
        dict = {
            'num_images': 0,
            'num_tokens': 0,
            'num_gen_tokens': 0
        }
        for req in self.batch:
            dict['num_images'] += req.num_images
            dict['num_tokens'] += req.num_context_tokens
            dict['num_gen_tokens'] += req.num_gen_tokens
        if self.batch.size == 0:
            #print('1')
            return
        costs = {
            'image_preproc' : (A * (dict['num_images']**0.5)) / self.batch.size,
            'prefil' : (B * (dict['num_tokens']**0.5)) / self.batch.size,
            'gen' : (C * (dict['num_gen_tokens']**0.5)) / self.batch.size
        }

        for req in self.batch:
            req.ticks['stage1'] = costs['image_preproc']
            req.ticks['stage2'] = costs['prefil']
            req.ticks['stage3'] = costs['gen']
            
            req.calc_t()
            req.calc_costs()

    def add_req(self, req):
        fp = req.get_footprint()
        self.batch = np.append(self.batch, req)
        self.curr_footprint += fp
        return self.status
    
    def get_status(self):
        return self.status
    
    def set_status(self, status):
        self.status = status

    def get_footprint(self):
        return self.curr_footprint
    
    def size(self):
        return self.batch.size
    
    def start(self):
        self.status = 0
        self.curr_footprint = 0
        self.batch = np.array([])

#----- Limitations------
P = 0 #time to first token - stages 1,2 - TTFT
D = 0 #time for any generated token

import copy

class Accelerator:
    def __init__(self):
        self.working = False
        self.stage = 'none'
        self.batch = Batch() 
        self.fl = 0
        self.idx = 0
        
    def start(self, batch):
        self.working = True
        self.batch = copy.deepcopy(batch)
        self.stage = 'stage1'
        self.size = batch.size()
        batch.start()

    def finish(self):
        self.working = False
        num = self.size
        self.stage = 'none'
        self.batch = Batch()
        self.fl = 0
        self.size = 0
        return num

    def is_working(self):
        return self.working
    
    def call(self, curr_time):
        tokens = 0
        ids = [(self.idx + i)%self.size for i in range(0, self.size)] 
        for i in ids:
            req = self.batch.batch[i]
        #for req in self.batch.batch:
            status = req.status
            if status == 'finished':
                continue
            if status == 'start':
                #print(self.size)
                # print(f'-----------\ncurr_time: {curr_time}\n')
                # req.print_req()
                # print(f'\n----------\n')
                req.set_status('stage1')
                status = 'stage1'

            if status == 'waiting' and self.fl==0:
                req.set_status(self.stage)
                status = self.stage
            
            if status != 'waiting':
                if req.ticks[status] != 0:
                    if tokens + req.costs[status] < K:
                        # print(f'--------\nstep\n')
                        # req.print_req()
                        # print(f'---------')            
                        tokens += req.costs[status] #1ms * 1/Bms = num_tokens per ms 
                        req.ticks[status] = max(0, req.ticks[status] - 1)
                    else:
                        #print('111111111111111111111111')
                        self.idx = i
                        return self.stage
                if req.ticks[status] == 0:
                    self.fl += 1
                    if self.fl != self.size:
                        req.set_status('waiting')
                        continue

            if status == 'stage1':
                if req.ticks['stage1'] == 0 and self.fl == self.size:
                    self.fl = 0
                    self.stage = 'stage2'
                    req.set_status('stage2')
                continue
            if status == 'stage2':
                if req.ticks['stage2'] == 0 and self.fl == self.size:
                    self.fl = 0
                    self.stage = 'stage3'
                    req.set_status('stage3')
                    req.end_stage_2 = curr_time
                    req.calc_ttft()
                continue
            if status == 'stage3':
                if req.ticks['stage3'] == 0 and self.fl == self.size:
                    req.set_status('finished')
                    self.fl = 0
                    self.stage = 'finished'
                    return self.stage
        return self.stage
            
from collections import deque
class ScheduleModel:
    def __init__(self, requests, end_time):
        self.curr_req = 0
        self.req_done = 0
        self.size = requests.size
        self.accelerators = [Accelerator() for _ in range(N)]
        self.end_time = end_time
        self.curr_time = 0
        self.Q = deque()
        self.requests = requests
        self.batch = Batch()
        self.deb = 0

    def tick(self):
        for a in self.accelerators:
            if a.is_working():
                if a.call(self.curr_time) =='finished':
                    #print('111111')
                    num = a.finish()
                    self.req_done += num
                    
                    print(self.req_done,'/',self.size)

                        #breakpoint()
                        #self.deb = 1

    def cycle(self):
        while self.curr_time < self.end_time+1:
            batch_status = self.batch.get_status() 
            
            while self.curr_req < self.size and self.requests[self.curr_req].get_offset() == self.curr_time:
                self.Q.append(self.requests[self.curr_req])
                #self.requests[self.curr_req].print_req()
                self.curr_req += 1
                if batch_status != 1: 
                    curr_elem = self.Q.popleft()
                    sum = self.batch.get_footprint() + curr_elem.get_footprint()
                    if sum < M:
                        self.batch.add_req(curr_elem)
                    else:
                        if self.batch.size() == 0:
                            print(sum)
                        self.batch.set_status(1)
                        self.Q.appendleft(curr_elem)
            
            if batch_status == 1:
                self.batch.create_agregation()
                
                for i in range(N):
                    if self.accelerators[i].is_working() == False:
                        self.accelerators[i].start(self.batch)
                        break
                
            self.tick()
            self.curr_time += 1
    
        print("!!!")
        print(len(self.Q))
        # for req in self.Q:
        #     req.print_req()
        while(len(self.Q)>0):
            if batch_status != 1: 
                curr_elem = self.Q.popleft()
                sum = self.batch.get_footprint() + curr_elem.get_footprint()
                if sum < M:
                    self.batch.add_req(curr_elem)

                else:
                    self.batch.set_status(1)
                    self.Q.appendleft(curr_elem)
                
            batch_status = self.batch.get_status() 
            if batch_status == 1:
                
                for i in range(N):
                    if self.accelerators[i].is_working() == False:
                        print(i)
                        self.accelerators[i].start(self.batch)
                        break
            self.tick()
            self.curr_time += 1

        while self.batch.size() != 0:
            for i in range(N):
                    if self.accelerators[i].is_working() == False:
                        print('starting')
                        self.accelerators[i].start(self.batch)
                        break
            self.tick()
            self.curr_time += 1
        
        while not all([self.accelerators[i].is_working() == False for i in range(N)]):
            self.tick()
            self.curr_time += 1

# test_df = pd.read_csv("test.csv", sep=';')
# test_df['TIMESTAMP'] = pd.to_datetime(test_df['TIMESTAMP'])
# test_df['offset_ms'] = (test_df['TIMESTAMP'] - test_df['TIMESTAMP'].min()).dt.total_seconds() * 1000
# test_df['offset_ms'] = test_df['offset_ms'].astype(int)

# start_time = 0
# end_time = test_df['offset_ms'].max()
# print(end_time)
# test_requests = np.array([Request(line) for _, line in test_df.iterrows()])
# s = test_requests.size
# print(s)
# model =  ScheduleModel(test_requests, end_time)
# model.cycle()

# ttft = [req.TTFT for req in test_requests]
# t = [req.T for req in test_requests]

df = pd.read_csv("AzureLMMInferenceTrace_multimodal.csv", sep=';')
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df['offset_ms'] = (df['TIMESTAMP'] - df['TIMESTAMP'].min()).dt.total_seconds() * 1000
df['offset_ms'] = df['offset_ms'].astype(int)

start_time = 0
end_time = df['offset_ms'].max()
print(end_time)
requests = np.array([Request(line) for _, line in df.iterrows()])
s = requests.size
print(s)
model =  ScheduleModel(requests, end_time)
model.cycle()

ttft = [req.TTFT for req in requests]
t = [req.T for req in requests]

if len(ttft) > 0:
    print(f"--- TTFT Statistics (ms) ---")
    print(f"Median: {np.median(ttft):.2f}")
    print(f"Average: {np.mean(ttft):.2f}")
    print(f"Min: {np.min(ttft):.2f}, Max: {np.max(ttft):.2f}")

if len(t) > 0:
    print(f"\n--- T (Time per Token) Statistics (ms/token) ---")
    print(f"Median: {np.median(t):.2f}")
    print(f"Average: {np.mean(t):.2f}")