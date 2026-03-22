#pragma once

#include <string>
#include <map>
#include <cstdint>
#include <iostream>

const double A = 100.0; // image preprocessing with certain cost of A
const double X = 2.0;   // image preprocessing with certain memory footprint of X
const double B = 100.0; // context token has associated cost of B
const double Y = 2.0;   // context token has associated footprint of Y
const double C = 100.0; // generated token costs C
const double Z = 2.0;   // generated token footprint of Z

const double EPS = 1e-6;

enum class Stage {
    NONE = 0,
    STAGE1 = 1,
    STAGE2 = 2,
    STAGE3 = 3,
    START = 4,
    WAITING = 5,
    FINISHED = 6
};

class Request {
public:
    Stage status;
    std::string timestep_str;
    int num_images;
    int num_context_tokens;
    int num_gen_tokens;
    int64_t offset_ms;
    int64_t end_stage_2;
    int64_t TTFT;
    double T;
    
    double ticks[4] = {0.0, 0.0, 0.0, 0.0}; 
    double costs[4] = {0.0, 0.0, 0.0, 0.0};
    double footprint;

    Request(std::string ts_str, int imgs, int ctx_tokens, int gen_tokens, int64_t offset) 
        : status(Stage::START), timestep_str(ts_str), num_images(imgs), 
          num_context_tokens(ctx_tokens), num_gen_tokens(gen_tokens), 
          offset_ms(offset), end_stage_2(0), TTFT(0), T(0) 
    {
        ticks[1] = A * num_images;
        ticks[2] = B * num_context_tokens;
        ticks[3] = C * num_gen_tokens;

        footprint = num_images * X + num_context_tokens * Y + num_gen_tokens * Z;
    }

    void calc_costs() {
        costs[1] = (ticks[1] < EPS) ? 0.0 : num_images / ticks[1];
        costs[2] = (ticks[2] < EPS) ? 0.0 : num_context_tokens / ticks[2];
        costs[3] = (ticks[3] < EPS) ? 0.0 : num_gen_tokens / ticks[3];
    }

    void set_status(Stage s) { status = s; }
    
    void calc_ttft() { TTFT = end_stage_2 - offset_ms; }

    void calc_t() { T = (num_gen_tokens == 0) ? 0.0 : ticks[3] / num_gen_tokens; }

    double get_footprint() const { return footprint; }
    
    int64_t get_offset() const { return offset_ms; }

    friend std::ostream& operator<<(std::ostream& os, const Request& req) {
        os << "Request with timestep " << req.timestep_str 
           << " has " << req.num_images << " images, " 
           << req.num_context_tokens << " text tokens and " 
           << req.num_gen_tokens << " gen tokens";
        return os;
    }
};