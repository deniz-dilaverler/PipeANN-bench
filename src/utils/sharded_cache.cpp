#include "utils/sharded_cache.hh"
#include <mutex>

template<typename T>
ShardedCache<T>::ShardedCache(size_t num_shards, size_t shard_capacity) : size(num_shards) {
  shards.reserve(num_shards);
  for (size_t i = 0; i < num_shards; i++) {
    shards.push_back(std::make_unique<Shard<T>>(shard_capacity));
  }
}

template<typename T>
void ShardedCache<T>::put(unsigned node_id, const std::shared_ptr<CacheNode<T>>& node) {
  size_t shard_id = this->get_shard_id(node_id);
  Shard<T>& shard = *shards[shard_id];
  std::lock_guard<std::mutex> shard_lock(shard.mut);

  auto it = shard.map.find(node_id);
  if (it != shard.map.end()) {
    // Erase from LRU list
    shard.lru.erase(it->second.second);
    // Push to the front of LRU list
    shard.lru.push_front(std::make_pair(node_id, node));
    // Update map
    it->second.first = node;
    it->second.second = shard.lru.begin();
  } else {
    // Evict if capacity reached
    if (shard.map.size() >= shard.capacity) {
      uint32_t lru_key = shard.lru.back().first;
      shard.lru.pop_back();
      shard.map.erase(lru_key);
    }
    // Insert new item
    shard.lru.push_front(std::make_pair(node_id, node));
    shard.map[node_id] = std::make_pair(node, shard.lru.begin());
  }
}

template<typename T>
bool ShardedCache<T>::get(uint32_t node_id, std::shared_ptr<CacheNode<T>>& node) {
  size_t shard_id = this->get_shard_id(node_id);
  Shard<T>& shard = *shards[shard_id];
  std::lock_guard<std::mutex> shard_lock(shard.mut);

  auto it = shard.map.find(node_id);
  if (it == shard.map.end()) {
    return false;
  }

  // Move element to front of LRU list
  shard.lru.splice(shard.lru.begin(), shard.lru, it->second.second);
  
  // Return value (shared_ptr copy)
  node = it->second.first;
  return true;
}

// Explicit template instantiations for supported types in PipeANN
template class ShardedCache<float>;
template class ShardedCache<int8_t>;
template class ShardedCache<uint8_t>;
