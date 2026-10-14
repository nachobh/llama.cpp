#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// Per-sequence acceptance controller. A full acceptance is a censored lower
// bound: probe upward instead of averaging it and getting stuck at short drafts.
struct common_speculative_adaptive {
    float acceptance = 2.0f;
    int32_t last_draft = 0;

    void reset() {
        acceptance = 2.0f;
        last_draft = 0;
    }

    int32_t limit(int32_t ceiling, int32_t minimum) {
        if (ceiling <= 0) {
            return ceiling;
        }
        acceptance = std::min(acceptance, float(ceiling));
        return std::min(ceiling, std::max(std::max(1, minimum), int32_t(std::lround(acceptance))));
    }

    void accept(uint16_t n_accepted) {
        if (last_draft <= 0) {
            return;
        }
        if (int32_t(n_accepted) >= last_draft) {
            acceptance += 1.0f;
        } else {
            acceptance = 0.75f * acceptance + 0.25f * float(n_accepted);
        }
        last_draft = 0;
    }
};
