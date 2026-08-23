#pragma once

#include "benchmark.hpp"
#include <map>
#include <string>

namespace bench {

    template <typename BenchMarkT>
    class BenchMarkParamsAggregator {
    public:
        struct Entry {
            std::function<BenchMarkT()> factory;
        };

        Result operator()() {
            Result last;
            for (auto& [name, entry] : benchmarks) {
                (void)name;
                last = entry.factory()();
            }
            return last;
        }

        std::map<std::string, Entry> benchmarks;
    };

} // namespace bench
