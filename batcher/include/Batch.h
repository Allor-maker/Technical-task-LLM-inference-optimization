#pragma once

#include <vector>
#include <memory>
#include "./Request.h"
#include <cmath>

const double P = 47750;

class Batch {
public:
    double curr_footprint;
    std::vector<std::shared_ptr<Request>> batch;
    int status;

    Batch() : curr_footprint(0.0), status(0) {}

    void create_agregation() {
        double dict_num_images = 0;
        double dict_num_tokens = 0;
        double dict_num_gen_tokens = 0;

        for (auto& req : batch) {
            dict_num_images += req->num_images;
            dict_num_tokens += req->num_context_tokens;
            dict_num_gen_tokens += req->num_gen_tokens;
        }

        if (batch.empty()) return;

        double costs_image_preproc = (A * std::sqrt(dict_num_images)) / batch.size();
        double costs_prefil = (B * std::sqrt(dict_num_tokens)) / batch.size();
        double costs_gen = (C * std::sqrt(dict_num_gen_tokens)) / batch.size();

        for (auto& req : batch) {
            req->ticks[1] = costs_image_preproc;
            req->ticks[2] = costs_prefil;
            req->ticks[3] = costs_gen;
            
            req->calc_t();
            req->calc_costs();
        }
    }
    void update_wait_t()
    {
        for (auto& req : batch)
        {
            req->wait_in_batch += 1;
            if (req->wait_in_batch >= P)
            {
                status = 2;
            }
        }
    }
    int add_req(std::shared_ptr<Request> req) {
        double fp = req->get_footprint();
        batch.push_back(req);
        curr_footprint += fp;
        return status;
    }
    
    int get_status() const { return status; }
    void set_status(int s) { status = s; }
    double get_footprint() const { return curr_footprint; }
    size_t size() const { return batch.size(); }
    
    void start() {
        status = 0;
        curr_footprint = 0.0;
        batch.clear();
    }
};