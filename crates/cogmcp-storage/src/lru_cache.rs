//! Thread-safe LRU (Least Recently Used) cache implementation
//!
//! This module provides a capacity-bounded cache that evicts the least recently
//! used entries when the cache is full. It supports optional TTL-based expiration
//! as a secondary eviction mechanism.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

/// A node in the doubly-linked list used for LRU ordering
struct LruNode<K, V> {
    #[allow(dead_code)]
    key: K,
    value: V,
    prev: Option<K>,
    next: Option<K>,
}

/// A thread-safe LRU cache with capacity-based eviction
///
/// This cache maintains entries in LRU order using a doubly-linked list.
/// When the cache reaches capacity, the least recently used entry is evicted.
/// Both `get()` and `insert()` operations update an entry's recency.
///
/// # Thread Safety
///
/// All operations are protected by a mutex, making this cache safe for
/// concurrent access from multiple threads.
///
/// # Example
///
/// ```
/// use cogmcp_storage::lru_cache::LruCache;
///
/// let cache = LruCache::new(2);
/// cache.insert("a", 1);
/// cache.insert("b", 2);
/// assert_eq!(cache.get(&"a"), Some(1));
///
/// // This evicts "b" since "a" was accessed more recently
/// cache.insert("c", 3);
/// assert_eq!(cache.get(&"b"), None);
/// assert_eq!(cache.get(&"c"), Some(3));
/// ```
pub struct LruCache<K, V> {
    inner: Mutex<LruCacheInner<K, V>>,
    capacity: usize,
}

struct LruCacheInner<K, V> {
    map: HashMap<K, LruNode<K, V>>,
    head: Option<K>, // Most recently used
    tail: Option<K>, // Least recently used
}

impl<K: Hash + Eq + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with the specified capacity
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LruCache capacity must be greater than 0");
        Self {
            inner: Mutex::new(LruCacheInner {
                map: HashMap::with_capacity(capacity),
                head: None,
                tail: None,
            }),
            capacity,
        }
    }

    /// Get a value from the cache, updating its recency
    ///
    /// Returns `Some(value)` if the key exists, `None` otherwise.
    /// Accessing a key makes it the most recently used.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        if inner.map.contains_key(key) {
            // Move to front (most recently used)
            Self::move_to_front(&mut inner, key);
            inner.map.get(key).map(|node| node.value.clone())
        } else {
            None
        }
    }

    /// Insert a key-value pair into the cache
    ///
    /// If the key already exists, the value is updated and the entry
    /// is moved to the front (most recently used position).
    ///
    /// If the cache is at capacity and the key is new, the least
    /// recently used entry is evicted before inserting.
    pub fn insert(&self, key: K, value: V) {
        let mut inner = self.inner.lock();

        if inner.map.contains_key(&key) {
            // Update existing entry
            if let Some(node) = inner.map.get_mut(&key) {
                node.value = value;
            }
            Self::move_to_front(&mut inner, &key);
        } else {
            // Evict if at capacity
            if inner.map.len() >= self.capacity {
                Self::evict_lru(&mut inner);
            }

            // Insert new entry at the front
            let old_head = inner.head.clone();
            let new_node = LruNode {
                key: key.clone(),
                value,
                prev: None,
                next: old_head.clone(),
            };

            // Update old head's prev pointer
            if let Some(ref old_head_key) = old_head {
                if let Some(old_head_node) = inner.map.get_mut(old_head_key) {
                    old_head_node.prev = Some(key.clone());
                }
            }

            inner.map.insert(key.clone(), new_node);

            // Update head pointer
            inner.head = Some(key.clone());

            // If this is the first entry, it's also the tail
            if inner.tail.is_none() {
                inner.tail = Some(key);
            }
        }
    }

    /// Remove a key from the cache
    ///
    /// Returns `Some(value)` if the key existed, `None` otherwise.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        Self::remove_from_list(&mut inner, key);
        inner.map.remove(key).map(|node| node.value)
    }

    /// Returns the number of entries in the cache
    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    /// Returns true if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.inner.lock().map.is_empty()
    }

    /// Returns the maximum capacity of the cache
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.map.clear();
        inner.head = None;
        inner.tail = None;
    }

    /// Move a key to the front of the LRU list (most recently used)
    fn move_to_front(inner: &mut LruCacheInner<K, V>, key: &K) {
        // If already at front, nothing to do
        if inner.head.as_ref() == Some(key) {
            return;
        }

        // Remove from current position
        Self::remove_from_list(inner, key);

        // Re-add at front
        if let Some(node) = inner.map.get_mut(key) {
            node.prev = None;
            node.next = inner.head.clone();
        }

        if let Some(ref old_head) = inner.head {
            if let Some(old_head_node) = inner.map.get_mut(old_head) {
                old_head_node.prev = Some(key.clone());
            }
        }

        inner.head = Some(key.clone());

        if inner.tail.is_none() {
            inner.tail = Some(key.clone());
        }
    }

    /// Remove a key from the doubly-linked list (but not from the map)
    fn remove_from_list(inner: &mut LruCacheInner<K, V>, key: &K) {
        let (prev, next) = match inner.map.get(key) {
            Some(node) => (node.prev.clone(), node.next.clone()),
            None => return,
        };

        // Update previous node's next pointer
        if let Some(ref prev_key) = prev {
            if let Some(prev_node) = inner.map.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        } else {
            // This was the head
            inner.head = next.clone();
        }

        // Update next node's prev pointer
        if let Some(ref next_key) = next {
            if let Some(next_node) = inner.map.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        } else {
            // This was the tail
            inner.tail = prev.clone();
        }
    }

    /// Evict the least recently used entry
    fn evict_lru(inner: &mut LruCacheInner<K, V>) {
        if let Some(tail_key) = inner.tail.clone() {
            Self::remove_from_list(inner, &tail_key);
            inner.map.remove(&tail_key);
        }
    }
}

/// A node in the LRU cache with TTL support
struct LruNodeWithTtl<K, V> {
    #[allow(dead_code)]
    key: K,
    value: V,
    prev: Option<K>,
    next: Option<K>,
    inserted_at: Instant,
}

/// A thread-safe LRU cache with both capacity-based and TTL-based eviction
///
/// This cache combines LRU eviction with time-based expiration. Entries are
/// evicted when either:
/// - The cache reaches capacity (LRU eviction)
/// - The entry's TTL has expired
///
/// # Example
///
/// ```
/// use cogmcp_storage::lru_cache::LruCacheWithTtl;
/// use std::time::Duration;
///
/// let cache = LruCacheWithTtl::new(100, Duration::from_secs(60));
/// cache.insert("key", "value");
/// assert_eq!(cache.get(&"key"), Some("value"));
/// ```
pub struct LruCacheWithTtl<K, V> {
    inner: Mutex<LruCacheWithTtlInner<K, V>>,
    capacity: usize,
    ttl: Duration,
}

struct LruCacheWithTtlInner<K, V> {
    map: HashMap<K, LruNodeWithTtl<K, V>>,
    head: Option<K>,
    tail: Option<K>,
}

impl<K: Hash + Eq + Clone, V: Clone> LruCacheWithTtl<K, V> {
    /// Create a new LRU cache with the specified capacity and TTL
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        assert!(capacity > 0, "LruCacheWithTtl capacity must be greater than 0");
        Self {
            inner: Mutex::new(LruCacheWithTtlInner {
                map: HashMap::with_capacity(capacity),
                head: None,
                tail: None,
            }),
            capacity,
            ttl,
        }
    }

    /// Get a value from the cache, updating its recency
    ///
    /// Returns `Some(value)` if the key exists and hasn't expired, `None` otherwise.
    /// Accessing a key makes it the most recently used.
    /// Expired entries are removed on access.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();

        // Check if key exists and is not expired
        let is_expired = inner.map.get(key).is_some_and(|node| node.inserted_at.elapsed() >= self.ttl);

        if is_expired {
            Self::remove_from_list(&mut inner, key);
            inner.map.remove(key);
            return None;
        }

        if inner.map.contains_key(key) {
            Self::move_to_front(&mut inner, key);
            inner.map.get(key).map(|node| node.value.clone())
        } else {
            None
        }
    }

    /// Insert a key-value pair into the cache
    ///
    /// If the key already exists, the value is updated, the TTL is reset,
    /// and the entry is moved to the front.
    ///
    /// If the cache is at capacity, the least recently used entry is evicted.
    pub fn insert(&self, key: K, value: V) {
        let mut inner = self.inner.lock();

        if inner.map.contains_key(&key) {
            // Update existing entry
            if let Some(node) = inner.map.get_mut(&key) {
                node.value = value;
                node.inserted_at = Instant::now();
            }
            Self::move_to_front(&mut inner, &key);
        } else {
            // Evict if at capacity
            if inner.map.len() >= self.capacity {
                Self::evict_lru(&mut inner);
            }

            let old_head = inner.head.clone();
            let new_node = LruNodeWithTtl {
                key: key.clone(),
                value,
                prev: None,
                next: old_head.clone(),
                inserted_at: Instant::now(),
            };

            if let Some(ref old_head_key) = old_head {
                if let Some(old_head_node) = inner.map.get_mut(old_head_key) {
                    old_head_node.prev = Some(key.clone());
                }
            }

            inner.map.insert(key.clone(), new_node);
            inner.head = Some(key.clone());

            if inner.tail.is_none() {
                inner.tail = Some(key);
            }
        }
    }

    /// Remove a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        Self::remove_from_list(&mut inner, key);
        inner.map.remove(key).map(|node| node.value)
    }

    /// Returns the number of entries in the cache (including potentially expired ones)
    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    /// Returns true if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.inner.lock().map.is_empty()
    }

    /// Returns the maximum capacity of the cache
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the TTL for entries in this cache
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.map.clear();
        inner.head = None;
        inner.tail = None;
    }

    /// Remove expired entries from the cache
    pub fn cleanup(&self) {
        let mut inner = self.inner.lock();
        let ttl = self.ttl;

        // Collect expired keys
        let expired_keys: Vec<K> = inner
            .map
            .iter()
            .filter(|(_, node)| node.inserted_at.elapsed() >= ttl)
            .map(|(k, _)| k.clone())
            .collect();

        // Remove expired entries
        for key in expired_keys {
            Self::remove_from_list(&mut inner, &key);
            inner.map.remove(&key);
        }
    }

    fn move_to_front(inner: &mut LruCacheWithTtlInner<K, V>, key: &K) {
        if inner.head.as_ref() == Some(key) {
            return;
        }

        Self::remove_from_list(inner, key);

        if let Some(node) = inner.map.get_mut(key) {
            node.prev = None;
            node.next = inner.head.clone();
        }

        if let Some(ref old_head) = inner.head {
            if let Some(old_head_node) = inner.map.get_mut(old_head) {
                old_head_node.prev = Some(key.clone());
            }
        }

        inner.head = Some(key.clone());

        if inner.tail.is_none() {
            inner.tail = Some(key.clone());
        }
    }

    fn remove_from_list(inner: &mut LruCacheWithTtlInner<K, V>, key: &K) {
        let (prev, next) = match inner.map.get(key) {
            Some(node) => (node.prev.clone(), node.next.clone()),
            None => return,
        };

        if let Some(ref prev_key) = prev {
            if let Some(prev_node) = inner.map.get_mut(prev_key) {
                prev_node.next = next.clone();
            }
        } else {
            inner.head = next.clone();
        }

        if let Some(ref next_key) = next {
            if let Some(next_node) = inner.map.get_mut(next_key) {
                next_node.prev = prev.clone();
            }
        } else {
            inner.tail = prev.clone();
        }
    }

    fn evict_lru(inner: &mut LruCacheWithTtlInner<K, V>) {
        if let Some(tail_key) = inner.tail.clone() {
            Self::remove_from_list(inner, &tail_key);
            inner.map.remove(&tail_key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_lru_cache_basic_insert_get() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        cache.insert("key1".to_string(), 1);
        cache.insert("key2".to_string(), 2);

        assert_eq!(cache.get(&"key1".to_string()), Some(1));
        assert_eq!(cache.get(&"key2".to_string()), Some(2));
        assert_eq!(cache.get(&"key3".to_string()), None);
    }

    #[test]
    fn test_lru_cache_update_value() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        cache.insert("key".to_string(), 1);
        assert_eq!(cache.get(&"key".to_string()), Some(1));

        cache.insert("key".to_string(), 2);
        assert_eq!(cache.get(&"key".to_string()), Some(2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let cache: LruCache<String, i32> = LruCache::new(3);
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.len(), 3);

        // Insert fourth item, should evict "a" (LRU)
        cache.insert("d".to_string(), 4);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"a".to_string()), None); // Evicted
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }

    #[test]
    fn test_lru_cache_access_updates_recency() {
        let cache: LruCache<String, i32> = LruCache::new(3);
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Access "a" to make it most recently used
        assert_eq!(cache.get(&"a".to_string()), Some(1));

        // Insert "d", should evict "b" (now LRU)
        cache.insert("d".to_string(), 4);

        assert_eq!(cache.get(&"a".to_string()), Some(1)); // Still there
        assert_eq!(cache.get(&"b".to_string()), None); // Evicted
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }

    #[test]
    fn test_lru_cache_remove() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        cache.insert("key".to_string(), 42);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(&"key".to_string());
        assert_eq!(removed, Some(42));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&"key".to_string()), None);
    }

    #[test]
    fn test_lru_cache_remove_nonexistent() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        let removed = cache.remove(&"nonexistent".to_string());
        assert_eq!(removed, None);
    }

    #[test]
    fn test_lru_cache_clear() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.get(&"a".to_string()), None);
    }

    #[test]
    fn test_lru_cache_capacity() {
        let cache: LruCache<String, i32> = LruCache::new(5);
        assert_eq!(cache.capacity(), 5);
    }

    #[test]
    fn test_lru_cache_single_item() {
        let cache: LruCache<String, i32> = LruCache::new(1);
        cache.insert("a".to_string(), 1);
        assert_eq!(cache.get(&"a".to_string()), Some(1));

        cache.insert("b".to_string(), 2);
        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_exactly_at_capacity() {
        let cache: LruCache<i32, i32> = LruCache::new(5);
        for i in 0..5 {
            cache.insert(i, i * 10);
        }

        assert_eq!(cache.len(), 5);
        for i in 0..5 {
            assert_eq!(cache.get(&i), Some(i * 10));
        }
    }

    #[test]
    fn test_lru_cache_empty() {
        let cache: LruCache<String, i32> = LruCache::new(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&"any".to_string()), None);
    }

    #[test]
    fn test_lru_cache_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(LruCache::new(100));
        let mut handles = vec![];

        // Spawn multiple threads that concurrently access the cache
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let key = format!("key-{}-{}", i, j);
                    cache_clone.insert(key.clone(), i * 100 + j);
                    cache_clone.get(&key);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should still be functional
        cache.insert("final".to_string(), 999);
        assert_eq!(cache.get(&"final".to_string()), Some(999));
    }

    #[test]
    #[should_panic(expected = "capacity must be greater than 0")]
    fn test_lru_cache_zero_capacity() {
        let _: LruCache<String, i32> = LruCache::new(0);
    }

    // Tests for LruCacheWithTtl

    #[test]
    fn test_lru_cache_with_ttl_basic() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(10, Duration::from_secs(60));
        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));
    }

    #[test]
    fn test_lru_cache_with_ttl_expiration() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(10, Duration::from_millis(50));
        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cache.get(&"key".to_string()), None);
    }

    #[test]
    fn test_lru_cache_with_ttl_eviction() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(2, Duration::from_secs(60));
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"a".to_string()), None); // Evicted
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_lru_cache_with_ttl_access_updates_recency() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(2, Duration::from_secs(60));
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        // Access "a" to make it most recently used
        cache.get(&"a".to_string());

        // Insert "c", should evict "b"
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_lru_cache_with_ttl_cleanup() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(10, Duration::from_millis(50));
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        thread::sleep(Duration::from_millis(100));

        // Insert a fresh entry
        cache.insert("c".to_string(), 3);

        // Run cleanup
        cache.cleanup();

        // Only "c" should remain
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_lru_cache_with_ttl_ttl_getter() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(10, Duration::from_secs(300));
        assert_eq!(cache.ttl(), Duration::from_secs(300));
    }

    #[test]
    fn test_lru_cache_with_ttl_insert_resets_ttl() {
        let cache: LruCacheWithTtl<String, i32> =
            LruCacheWithTtl::new(10, Duration::from_millis(100));
        cache.insert("key".to_string(), 1);

        thread::sleep(Duration::from_millis(60));

        // Re-insert to reset TTL
        cache.insert("key".to_string(), 2);

        thread::sleep(Duration::from_millis(60));

        // Should still be valid because TTL was reset
        assert_eq!(cache.get(&"key".to_string()), Some(2));
    }
}
