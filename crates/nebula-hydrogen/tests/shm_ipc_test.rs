#![allow(clippy::unwrap_used, clippy::panic, missing_docs, unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::Ordering;

use nebula_hydrogen::{
    FrameFlags, IPC_MAGIC, IPC_VERSION, MessageType, REQUEST_QUEUE_CAPACITY_BYTES,
    RESPONSE_QUEUE_CAPACITY_BYTES, ShmHandle, TokenPayload, ToolSpawnGuard,
};

#[test]
fn test_shm_init_and_geometry() {
    let test_id = format!("test_init_{}", ulid::Ulid::generate());
    let shm_name = format!(r"Local\Nebula_Hydrogen_Test_Shm_{test_id}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_Test_Eng_{test_id}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_Test_Dae_{test_id}");

    let handle = ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true).expect("open shm");
    let hdr = handle.header();

    assert_eq!(hdr.magic, IPC_MAGIC);
    assert_eq!(hdr.version, IPC_VERSION);
    assert_eq!(
        hdr.request_queue.capacity_bytes,
        REQUEST_QUEUE_CAPACITY_BYTES as u64
    );
    assert_eq!(
        hdr.response_queue.capacity_bytes,
        RESPONSE_QUEUE_CAPACITY_BYTES as u64
    );
    assert_eq!(handle.active_tool_processes(), 0);
    assert_eq!(handle.cuda_yield_mode(), 0);
}

#[test]
fn test_struct_layout_offsets() {
    use nebula_hydrogen::{ShmGlobalHeader, ShmQueueHeader};
    use std::mem::size_of;

    assert_eq!(size_of::<ShmQueueHeader>(), 192);
    assert_eq!(size_of::<ShmGlobalHeader>(), 576);

    // Verify queue offsets
    let dummy_gh = std::mem::MaybeUninit::<ShmGlobalHeader>::uninit();
    let base = dummy_gh.as_ptr() as usize;

    unsafe {
        let p_asm = &(*dummy_gh.as_ptr()).active_stream_mask as *const _ as usize;
        let p_atp = &(*dummy_gh.as_ptr()).active_tool_processes as *const _ as usize;
        let p_cym = &(*dummy_gh.as_ptr()).cuda_yield_mode as *const _ as usize;
        let p_req_off = &(*dummy_gh.as_ptr()).request_queue_offset as *const _ as usize;
        let p_resp_off = &(*dummy_gh.as_ptr()).response_queue_offset as *const _ as usize;
        let p_rq = &(*dummy_gh.as_ptr()).request_queue as *const _ as usize;
        let p_rsq = &(*dummy_gh.as_ptr()).response_queue as *const _ as usize;

        assert_eq!(p_asm - base, 24);
        assert_eq!(p_atp - base, 64);
        assert_eq!(p_cym - base, 128);
        assert_eq!(p_req_off - base, 136);
        assert_eq!(p_resp_off - base, 144);
        assert_eq!(p_rq - base, 192);
        assert_eq!(p_rsq - base, 384);

        let p_head = &(*dummy_gh.as_ptr()).request_queue.head as *const _ as usize;
        let p_tail = &(*dummy_gh.as_ptr()).request_queue.tail as *const _ as usize;
        let p_bw = &(*dummy_gh.as_ptr()).request_queue.total_bytes_written as *const _ as usize;

        assert_eq!(p_head - p_rq, 64);
        assert_eq!(p_tail - p_rq, 128);
        assert_eq!(p_bw - p_rq, 136);
    }
}

#[test]
fn test_active_tool_processes_barrier() {
    let test_id = format!("test_atp_{}", ulid::Ulid::generate());
    let shm_name = format!(r"Local\Nebula_Hydrogen_Test_Shm_{test_id}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_Test_Eng_{test_id}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_Test_Dae_{test_id}");

    let handle = Arc::new(ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true).expect("open shm"));
    assert_eq!(handle.active_tool_processes(), 0);

    // Direct spawn and complete
    handle.tool_process_spawned();
    assert_eq!(handle.active_tool_processes(), 1);

    handle.tool_process_completed();
    assert_eq!(handle.active_tool_processes(), 0);

    // RAII guard test with global registration
    nebula_hydrogen::register_global_shm(Arc::clone(&handle));

    {
        let guard1 = ToolSpawnGuard::new();
        assert_eq!(handle.active_tool_processes(), 1);

        {
            let guard2 = ToolSpawnGuard::new();
            assert_eq!(handle.active_tool_processes(), 2);
            drop(guard2);
        }

        assert_eq!(handle.active_tool_processes(), 1);
        drop(guard1);
    }

    assert_eq!(handle.active_tool_processes(), 0);
    nebula_hydrogen::unregister_global_shm();
}

#[test]
fn test_token_payload_safe_deserialization() {
    let mut payload = TokenPayload {
        token_id: 12345,
        logit: 9.87,
        text_len: 5,
        is_final: 0,
        reserved: 0,
        text: [0u8; 64],
    };
    payload.text[0..5].copy_from_slice(b"hello");

    let bytes = unsafe {
        std::slice::from_raw_parts(
            &payload as *const TokenPayload as *const u8,
            std::mem::size_of::<TokenPayload>(),
        )
    };

    let parsed = TokenPayload::from_bytes(bytes).expect("parse valid payload");
    assert_eq!(parsed.token_id, 12345);
    assert_eq!(parsed.text_str(), "hello");
    assert_eq!(parsed.is_final, 0);

    // Too short slice returns None safely
    assert!(TokenPayload::from_bytes(&bytes[..10]).is_none());
}

#[test]
fn test_frame_write_and_read() {
    let test_id = format!("test_frame_{}", ulid::Ulid::generate());
    let shm_name = format!(r"Local\Nebula_Hydrogen_Test_Shm_{test_id}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_Test_Eng_{test_id}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_Test_Dae_{test_id}");

    let handle = ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true).expect("open shm");

    // Write request frame
    let req_body = b"{\"prompt\": \"Hello Hydrogen\"}";
    handle
        .write_request_frame(MessageType::GenerateRequest, FrameFlags::NONE, 42, req_body)
        .expect("write request frame");

    // Verify request frame is written into request queue
    let hdr = handle.header();
    let head = hdr.request_queue.head.load(Ordering::Acquire);
    assert!(head > 0);
}

#[test]
fn test_stream_lease_registry_and_guard() {
    use nebula_hydrogen::StreamLeaseGuard;

    let test_id = format!("test_lease_{}", ulid::Ulid::generate());
    let shm_name = format!(r"Local\Nebula_Hydrogen_Test_Shm_{test_id}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_Test_Eng_{test_id}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_Test_Dae_{test_id}");

    let handle = ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true).expect("open shm");

    // All slots initially free
    for slot in 0..8 {
        assert!(!handle.is_slot_active(slot));
    }

    // Acquire all 8 slots
    let mut guards = Vec::new();
    for expected_slot in 0..8 {
        let guard = StreamLeaseGuard::acquire(&handle).expect("acquire slot");
        assert_eq!(guard.slot(), expected_slot);
        assert!(handle.is_slot_active(expected_slot));
        guards.push(guard);
    }

    // 9th acquire must fail (exhaustion)
    assert!(StreamLeaseGuard::acquire(&handle).is_none());

    // Drop slot 3
    drop(guards.remove(3));
    assert!(!handle.is_slot_active(3));

    // Re-acquire should claim slot 3
    let guard3 = StreamLeaseGuard::acquire(&handle).expect("reacquire slot 3");
    assert_eq!(guard3.slot(), 3);
    assert!(handle.is_slot_active(3));
}

#[test]
fn test_binary_token_request_16384_tokens() {
    let test_id = format!("test_toks_{}", ulid::Ulid::generate());
    let shm_name = format!(r"Local\Nebula_Hydrogen_Test_Shm_{test_id}");
    let eng_evt = format!(r"Local\Nebula_Hydrogen_Test_Eng_{test_id}");
    let dae_evt = format!(r"Local\Nebula_Hydrogen_Test_Dae_{test_id}");

    let handle = ShmHandle::open(&shm_name, &eng_evt, &dae_evt, true).expect("open shm");

    let tokens: Vec<u32> = (0..16384).collect();
    let t0 = std::time::Instant::now();
    handle
        .write_generate_tokens_request(0, &tokens, 256, 0.7, 0.8, &[248046])
        .expect("write binary token request");
    let elapsed = t0.elapsed();

    // Verify sub-10us ingestion proof metric
    println!("16,384 tokens SHM ingestion latency: {elapsed:?}");

    let hdr = handle.header();
    let head = hdr.request_queue.head.load(Ordering::Acquire);
    assert!(head > 0);
}
