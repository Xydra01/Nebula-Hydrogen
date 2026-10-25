//! Bounded ring buffer containing unsafe pointer logic to refactor.

pub struct BoundedRingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize,
    tail: usize,
    len: usize,
    cap: usize,
}

impl<T> BoundedRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        let mut buf = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buf.push(None);
        }
        Self {
            buf,
            head: 0,
            tail: 0,
            len: 0,
            cap: capacity,
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.len >= self.cap {
            return Err(item);
        }
        // Legacy: uses raw unsafe pointer access to be refactored to safe Rust
        unsafe {
            let ptr = self.buf.as_mut_ptr().add(self.tail);
            *ptr = Some(item);
        }
        self.tail = (self.tail + 1) % self.cap;
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // Legacy: uses raw unsafe pointer access to be refactored to safe Rust
        let val = unsafe {
            let ptr = self.buf.as_mut_ptr().add(self.head);
            (*ptr).take()
        };
        self.head = (self.head + 1) % self.cap;
        self.len -= 1;
        val
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
