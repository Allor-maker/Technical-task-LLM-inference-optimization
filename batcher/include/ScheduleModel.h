#pragma once

#include <deque>

#include "./Accelerator.h"

const int N = 10;           // num accelerators
const double M = 317000.0;  // available memory
double max_footprint = -1.0;
class ScheduleModel {
public:
    size_t curr_req;
    size_t req_done;
    size_t total_size;
    std::vector<Accelerator> accelerators;
    int64_t end_time;
    int64_t curr_time;
    std::deque<std::shared_ptr<Request>> Q;
    std::vector<std::shared_ptr<Request>> requests;
    Batch batch;
    int deb;

    ScheduleModel(std::vector<std::shared_ptr<Request>> reqs, int64_t et)
        : curr_req(0), req_done(0), total_size(reqs.size()), 
          accelerators(N), end_time(et), curr_time(0), 
          requests(reqs), deb(0) {}

    void tick() {
        for (auto& a : accelerators) {
            if (a.is_working()) {
                if (a.call(curr_time) == Stage::FINISHED) {
                    size_t num = a.finish();
                    req_done += num;
                    std::cout << req_done << " / " << total_size << "\n";
                }
            }
        }
    }

    void cycle() {
        while (curr_time < end_time + 1) {
            int batch_status = batch.get_status();

            while (curr_req < total_size && requests[curr_req]->get_offset() == curr_time) {
                Q.push_back(requests[curr_req]);
                curr_req++;
            }
            if (!Q.empty() && batch_status != 1)
            {
                auto curr_elem = Q.front();
                Q.pop_front();
                double sum = batch.get_footprint() + curr_elem->get_footprint();
                if (sum < M) {
                    batch.add_req(curr_elem);
                    
                } else {
                    if (batch.size() == 0) std::cout << sum << "\n";
                    batch.set_status(1);
                    Q.push_front(curr_elem);
                }
            }
            
            
            if (batch_status == 1) {
                batch.create_agregation();
                for (int i = 0; i < N; ++i) {
                    if (!accelerators[i].is_working()) {
                        accelerators[i].start(batch);
                        break;
                    }
                }
            }

            if (batch_status == 2)
            {
                
                batch_status = 1;
                batch.create_agregation();
                for (int i = 0; i < N; ++i) {
                    if (!accelerators[i].is_working()) {
                        accelerators[i].start(batch);
                        break;
                    }
                }
            }
            tick();
            batch.update_wait_t();
            curr_time++;
        }

        while (!Q.empty()) {
            if (batch.get_status() != 1) {
                auto curr_elem = Q.front();
                Q.pop_front();
                double curr_fp = curr_elem->get_footprint();
                if (curr_fp > max_footprint)
                    max_footprint = curr_fp;
                double sum = batch.get_footprint() + curr_fp;
                
                if (sum < M) {
                    batch.add_req(curr_elem);
                } else {
                    batch.set_status(1);
                    Q.push_front(curr_elem);
                }
            }
                
            int batch_status = batch.get_status();
            if (batch_status == 1) {
                for (int i = 0; i < N; ++i) {
                    if (!accelerators[i].is_working()) {
                        accelerators[i].start(batch);
                        break;
                    }
                }
            }
            tick();
            curr_time++;
        }

        while (batch.size() != 0) {
            for (int i = 0; i < N; ++i) {
                if (!accelerators[i].is_working()) {
                    accelerators[i].start(batch);
                    break;
                }
            }
            tick();
            curr_time++;
        }
        
        bool all_idle = false;
        while (!all_idle) {
            all_idle = true;
            for (int i = 0; i < N; ++i) {
                if (accelerators[i].is_working()) {
                    all_idle = false;
                    break;
                }
            }
            if (!all_idle) {
                tick();
                curr_time++;
            }
        }
        
    }
};