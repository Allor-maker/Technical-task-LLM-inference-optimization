#pragma once

#include <cstddef>
#include <numeric>

#include "./Batch.h"

const double K = 1000.0;    // computing capability

class Accelerator {
public:
    bool working;
    Stage stage;
    Batch batch;
    int fl;
    size_t idx;
    size_t current_size;

    Accelerator() : working(false), stage(Stage::NONE), fl(0), idx(0), current_size(0) {}

    void start(Batch& b) {
        working = true;
        batch = b;
        stage = Stage::STAGE1;;
        current_size = b.size();
        b.start();
    }

    size_t finish() {
        working = false;
        size_t num = current_size;
        stage = Stage::NONE;
        batch = Batch();
        fl = 0;
        current_size = 0;
        return num;
    }

    bool is_working() const { return working; }
    
    Stage call(int64_t curr_time) {
        double tokens = 0.0;
        
        std::vector<size_t> ids;
        for (size_t i = 0; i < current_size; ++i) {
            ids.push_back((idx + i) % current_size);
        }

        for (size_t i : ids) {
            auto req = batch.batch[i];
            Stage status = req->status;

            if (status == Stage::FINISHED) continue;

            if (status == Stage::START) {
                req->set_status(Stage::STAGE1);
                status = Stage::STAGE1;
            }

            if (status == Stage::WAITING && fl == 0) {
                req->set_status(stage);
                status = stage;
            }
            
            if (status != Stage::WAITING) {
                int s_idx = static_cast<int>(status);
                if (req->ticks[s_idx] != 0.0) {
                    if (tokens + req->costs[s_idx] < K) {
                        tokens += req->costs[s_idx];
                        req->ticks[s_idx] = std::max(0.0, req->ticks[s_idx] - 1.0);
                    } else {
                        idx = i;
                        return stage;
                    }
                }
                if (req->ticks[s_idx] == 0.0) {
                    fl += 1;
                    if (fl != (int)current_size) {
                        req->set_status(Stage::WAITING);
                        continue;
                    }
                }
            }

            if (status == Stage::STAGE1) {
                if (req->ticks[1] == 0.0 && fl == (int)current_size) {
                    fl = 0;
                    stage = Stage::STAGE2;
                    req->set_status(Stage::STAGE2);
                }
                continue;
            }
            if (status == Stage::STAGE2) {
                if (req->ticks[2] == 0.0 && fl == (int)current_size) {
                    fl = 0;
                    stage = Stage::STAGE3;
                    req->set_status(Stage::STAGE3);
                    req->end_stage_2 = curr_time;
                    req->calc_ttft();
                }
                continue;
            }
            if (status == Stage::STAGE3) {
                if (req->ticks[3] == 0.0 && fl == (int)current_size) {
                    req->set_status(Stage::FINISHED);
                    fl = 0;
                    stage = Stage::FINISHED;
                    return stage;
                }
            }
        }
        return stage;
    }
};