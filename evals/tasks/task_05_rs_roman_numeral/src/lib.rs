//! Roman numeral conversion with non-canonical rejection bug.

/// Parse standard Roman numerals between 1 and 3999.
pub fn roman_to_int(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }

    let val = |c| match c {
        'I' => Some(1),
        'V' => Some(5),
        'X' => Some(10),
        'L' => Some(50),
        'C' => Some(100),
        'D' => Some(500),
        'M' => Some(1000),
        _ => None,
    };

    let mut total = 0;
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        let v = val(chars[i])?;
        if i + 1 < chars.len() && v < val(chars[i + 1])? {
            total -= v;
        } else {
            total += v;
        }
    }

    // BUG: does not validate canonical reconstructibility (e.g. "IIII", "VX", "IC", "MMMM")
    if (1..=3999).contains(&total) {
        Some(total as u32)
    } else {
        None
    }
}
