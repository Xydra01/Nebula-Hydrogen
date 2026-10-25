use task_05_roman::roman_to_int;

#[test]
fn test_valid_roman_numerals() {
    assert_eq!(roman_to_int("III"), Some(3));
    assert_eq!(roman_to_int("LVIII"), Some(58));
    assert_eq!(roman_to_int("MCMXCIV"), Some(1994));
    assert_eq!(roman_to_int("MMMCMXCIX"), Some(3999));
    assert_eq!(roman_to_int("XLII"), Some(42));
}

#[test]
fn test_invalid_roman_numerals() {
    for bad in ["", "IIII", "VX", "IC", "MMMM", "ABC", "IIV", "VV", "XM", "iv"] {
        assert_eq!(roman_to_int(bad), None, "rejected {bad}");
    }
}
