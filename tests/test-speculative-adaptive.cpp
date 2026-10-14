#include "../common/speculative-adaptive.h"

#include <cstdlib>
#include <iostream>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int main() {
    common_speculative_adaptive controller;
    check(controller.limit(7, 0) == 2, "initial draft");
    for (int step = 0; step < 20; ++step) {
        controller.last_draft = controller.limit(7, 0);
        controller.accept(uint16_t(controller.last_draft));
    }
    check(controller.limit(7, 0) == 7, "full acceptance must probe to ceiling");
    for (int step = 0; step < 20; ++step) {
        controller.last_draft = controller.limit(7, 0);
        controller.accept(0);
    }
    check(controller.limit(7, 0) == 1, "partial acceptance backs off");
    check(controller.limit(7, 3) == 3, "minimum draft");
    check(controller.limit(2, 3) == 2, "caller ceiling wins over minimum");
    check(controller.limit(0, 3) == 0, "zero caller budget");
    controller.reset();
    check(controller.limit(7, 0) == 2 && controller.last_draft == 0, "new prompt resets state");
    controller.last_draft = 2;
    controller.accept(2);
    check(controller.limit(7, 0) == 3, "one clean draft probes upward");
    const float before = controller.acceptance;
    controller.accept(0);
    check(controller.acceptance == before, "duplicate accept without draft is ignored");
    common_speculative_adaptive other_sequence;
    check(other_sequence.limit(7, 0) == 2, "sequence state is isolated");
    std::cout << "adaptive draft controller: OK\n";
}
