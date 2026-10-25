use task_07_ring::BoundedRingBuffer;

#[test]
fn test_ring_fifo() {
    let mut ring = BoundedRingBuffer::new(3);
    assert!(ring.is_empty());
    assert_eq!(ring.push(10), Ok(()));
    assert_eq!(ring.push(20), Ok(()));
    assert_eq!(ring.push(30), Ok(()));
    assert_eq!(ring.push(40), Err(40));
    assert_eq!(ring.len(), 3);

    assert_eq!(ring.pop(), Some(10));
    assert_eq!(ring.push(50), Ok(()));
    assert_eq!(ring.pop(), Some(20));
    assert_eq!(ring.pop(), Some(30));
    assert_eq!(ring.pop(), Some(50));
    assert_eq!(ring.pop(), None);
}
