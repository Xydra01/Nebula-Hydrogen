#pragma once

#include <cstddef>
#include <optional>
#include <span>

class BufferView {
public:
    BufferView(const char* data, std::size_t size);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::optional<BufferView> subview(std::size_t offset, std::size_t count) const;
    [[nodiscard]] const char* data() const noexcept;

private:
    std::span<const char> span_;
};
