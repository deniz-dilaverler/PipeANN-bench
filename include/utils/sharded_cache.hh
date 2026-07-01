#ifndef SHARDED_CACHE_HH_
#define SHARDED_CACHE_HH_
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

template<typename T>
struct CacheNode {
  std::vector<T> coords;
  std::vector<uint32_t> nbrs;
  std::vector<char> labels;
};

template<typename T>
struct Shard {
  std::mutex mut;
  std::list<std::pair<uint32_t, std::shared_ptr<CacheNode<T>>>> lru;
  size_t capacity;
  std::unordered_map<uint32_t, std::pair<std::shared_ptr<CacheNode<T>>, typename std::list<std::pair<uint32_t, std::shared_ptr<CacheNode<T>>>>::iterator>> map;
  Shard(size_t cap = 1000) : capacity(cap) {};
};

template<typename T>
class ShardedCache {
 private:
  size_t size;
  std::vector<Shard<T>> shards;

 public:
  ShardedCache(size_t num_shards, size_t shard_capacity);
  void put(unsigned node_id, const std::shared_ptr<CacheNode<T>>& node);
  bool get(uint32_t node_id, std::shared_ptr<CacheNode<T>>& node);

 private:
  inline size_t get_shard_id(unsigned node_id) const {
    return node_id % size;
  }
};

#endif
