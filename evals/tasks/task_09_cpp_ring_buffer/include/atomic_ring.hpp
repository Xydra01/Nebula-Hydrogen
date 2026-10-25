#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

template <typename T>
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(std::size_t capacity)
        : capacity_(capacity), buffer_(capacity) {}

    bool push(const T& item) {
        // Starter stub: needs atomic head/tail with capacity check
        (void)item;
        return false;
    }

    std::optional<T> pop() {
        // Starter stub
        return std::nullopt;
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }

private:
    std::size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};
