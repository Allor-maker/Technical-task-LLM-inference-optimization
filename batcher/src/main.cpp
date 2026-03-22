#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "../include/Request.h"
#include "../include/Batch.h"
#include "../include/ScheduleModel.h"

int64_t parse_time_to_ms(const std::string& datetime) {
    int year = 0, month = 0, day = 0, hour = 0, min = 0;
    double sec_double = 0.0;

    sscanf(datetime.c_str(), "%d-%d-%dT%d:%d:%lf",
        &year, &month, &day, &hour, &min, &sec_double);

    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = min;
    tm.tm_sec = static_cast<int>(sec_double);
    tm.tm_isdst = -1;

    time_t time_seconds = std::mktime(&tm);

    int ms = static_cast<int>(std::round((sec_double - tm.tm_sec) * 1000.0));

    return static_cast<int64_t>(time_seconds) * 1000 + ms;
}

double get_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
    return v[n/2];
}

double get_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

int main() {
    std::ifstream file("data/AzureLMMInferenceTrace_multimodal.csv");
    if (!file.is_open()) {
        std::cerr << "Cannot open file!" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line);

    struct RawRow {
        std::string timestamp;
        int num_images;
        int ctx_tokens;
        int gen_tokens;
        int64_t raw_ms;
    };

    std::vector<RawRow> raw_data;
    int64_t min_ms = -1;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string ts, ni_str, ct_str, gt_str;
        
        std::getline(ss, ts, ';');
        std::getline(ss, ni_str, ';');
        std::getline(ss, ct_str, ';');
        std::getline(ss, gt_str, ';');

        int64_t ms = parse_time_to_ms(ts);
        if (min_ms == -1 || ms < min_ms) {
            min_ms = ms;
        }

        raw_data.push_back({ts, std::stoi(ni_str), std::stoi(ct_str), std::stoi(gt_str), ms});
    }

    std::vector<std::shared_ptr<Request>> requests;
    int64_t end_time = 0;

    for (const auto& row : raw_data) {
        int64_t offset = row.raw_ms - min_ms;
        if (offset > end_time) end_time = offset;

        requests.push_back(std::make_shared<Request>(
            row.timestamp, row.num_images, row.ctx_tokens, row.gen_tokens, offset
        ));
    }

    std::cout << end_time << "\n";
    std::cout << requests.size() << "\n";

    ScheduleModel model(requests, end_time);
    model.cycle();

    std::vector<double> ttft, t_stats;
    for (const auto& req : requests) {
        ttft.push_back(static_cast<double>(req->TTFT));
        t_stats.push_back(req->T);
    }

    if (!ttft.empty()) {
        double min_val = *std::min_element(ttft.begin(), ttft.end());
        double max_val = *std::max_element(ttft.begin(), ttft.end());
        
        std::cout << "--- TTFT Statistics (ms) ---\n";
        std::cout << "Median: " << std::fixed << std::setprecision(2) << get_median(ttft) << "\n";
        std::cout << "Average: " << get_mean(ttft) << "\n";
        std::cout << "Min: " << min_val << ", Max: " << max_val << "\n";
    }

    if (!t_stats.empty()) {
        std::cout << "\n--- T (Time per Token) Statistics (ms/token) ---\n";
        std::cout << "Median: " << std::fixed << std::setprecision(2) << get_median(t_stats) << "\n";
        std::cout << "Average: " << get_mean(t_stats) << "\n";
    }

    return 0;
}