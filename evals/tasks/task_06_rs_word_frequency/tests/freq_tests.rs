use task_06_freq::word_freq;

#[test]
fn test_word_frequencies() {
    assert_eq!(word_freq(""), vec![]);
    assert_eq!(
        word_freq("The, the... the!"),
        vec![("the".to_string(), 3)]
    );
    assert_eq!(
        word_freq("b a b a c"),
        vec![
            ("a".to_string(), 2),
            ("b".to_string(), 2),
            ("c".to_string(), 1),
        ]
    );
}
