#include "buffer_view.hpp"
#include <cassert>
#include <string_view>

int main() {
    std::string_view text = "Hello, World!";
    BufferView bv(text.data(), text.size());

    assert(bv.size() == 13);
    assert(!bv.empty());

    // Exact subview
    auto full = bv.subview(0, 13);
    assert(full.has_value());
    assert(full->size() == 13);

    // Partial subview
    auto part = bv.subview(7, 5);
    assert(part.has_value());
    assert(part->size() == 5);

    // Out of bounds
    auto bad = bv.subview(10, 10);
    assert(!bad.has_value());

    return 0;
}
