#include "buffer_view.hpp"

BufferView::BufferView(const char* data, std::size_t size)
    : span_(data, size) {}

std::size_t BufferView::size() const noexcept {
    return span_.size();
}

bool BufferView::empty() const noexcept {
    return span_.empty();
}

const char* BufferView::data() const noexcept {
    return span_.data();
}

std::optional<BufferView> BufferView::subview(std::size_t offset, std::size_t count) const {
    // BUG: strict >= prevents taking exact full subview where offset + count == size()
    if (offset + count >= span_.size()) {
        return std::nullopt;
    }
    return BufferView(span_.data() + offset, count);
}
