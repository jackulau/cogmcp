//! In-memory cache for frequently accessed data

use dashmap::DashMap;
use std::time::{Duration, Instant};

/// A simple time-based cache
pub struct Cache<K, V> {
    data: DashMap<K, CacheEntry<V>>,
    ttl: Duration,
}

struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
}

impl<K: std::hash::Hash + Eq, V: Clone> Cache<K, V> {
    /// Create a new cache with the given TTL
    pub fn new(ttl: Duration) -> Self {
        Self {
            data: DashMap::new(),
            ttl,
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.data.get(key).and_then(|entry| {
            if entry.inserted_at.elapsed() < self.ttl {
                Some(entry.value.clone())
            } else {
                None
            }
        })
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        self.data.insert(
            key,
            CacheEntry {
                value,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &K) {
        self.data.remove(key);
    }

    /// Clear all expired entries
    pub fn cleanup(&self) {
        let ttl = self.ttl;
        self.data.retain(|_, entry| entry.inserted_at.elapsed() < ttl);
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_basic() {
        let cache: Cache<String, i32> = Cache::new(Duration::from_secs(60));
        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));
    }

    #[test]
    fn test_cache_expiry() {
        let cache: Cache<String, i32> = Cache::new(Duration::from_millis(50));
        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(42));

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cache.get(&"key".to_string()), None);
    }
}
