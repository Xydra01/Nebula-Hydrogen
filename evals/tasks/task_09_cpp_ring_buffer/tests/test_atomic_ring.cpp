#include "atomic_ring.hpp"
#include <cassert>

int main() {
    SpscRingBuffer<int> ring(4);
    assert(ring.capacity() == 4);

    // After implementation:
    // assert(ring.push(1));
    // assert(ring.push(2));
    // auto v = ring.pop();
    // assert(v.has_value() && *v == 1);

    return 0;
}
