//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once
#include <atomic>
#include <memory>
#include <utility>

#include "memory/allocator.h"
#include "memory/arena.h"
#include "port/lang.h"
#include "port/likely.h"
#include "util/core_local.h"
#include "util/mutexlock.h"
#include "util/thread_local.h"

// Only generate field unused warning for padding array, or build under
// GCC 4.8.1 will fail.
#ifdef __clang__
#define ROCKSDB_FIELD_UNUSED __attribute__((__unused__))
#else
#define ROCKSDB_FIELD_UNUSED
#endif  // __clang__

namespace ROCKSDB_NAMESPACE {

class Logger;

/**
 * Allocator
 * 抽象基类，定义了内存分配接口（如 Allocate、AllocateAligned），不关心具体实现和线程安全。

 * Arena
 * 继承自 Allocator，实现了高效的顺序内存分配，适合单线程或低并发场景。负责实际的内存块管理和分配逻辑。

 * ConcurrentArena
 * 继承自 Allocator，内部组合（包含）一个 Arena 实例。扩展了线程安全和多核分片缓存功能，适合高并发场景。所有分配最终都委托给 Arena 完成。

 *Arena 和 ConcurrentArena 都实现了 Allocator 的接口。
 * ConcurrentArena 内部实际使用 Arena 进行内存分配。

 * Arena 只适合单线程，ConcurrentArena 适合多线程。
 * ConcurrentArena 增加了分片缓存和锁机制，减少多线程争用，提高分配效率。
 */

// ConcurrentArena wraps an Arena.  It makes it thread safe using a fast
// inlined spinlock, and adds small per-core allocation caches to avoid
// contention for small allocations.  To avoid any memory waste from the
// per-core shards, they are kept small, they are lazily instantiated
// only if ConcurrentArena actually notices concurrent use, and they
// adjust their size so that there is no fragmentation waste when the
// shard blocks are allocated from the underlying main arena.
class ConcurrentArena : public Allocator {
 public:
  // block_size and huge_page_size are the same as for Arena (and are
  // in fact just passed to the constructor of arena_.  The core-local
  // shards compute their shard_block_size as a fraction of block_size
  // that varies according to the hardware concurrency level.
  explicit ConcurrentArena(size_t block_size = Arena::kMinBlockSize,
                           AllocTracker* tracker = nullptr,
                           size_t huge_page_size = 0);

  char* Allocate(size_t bytes) override {
    return AllocateImpl(bytes, false /*force_arena*/,
                        [this, bytes]() { return arena_.Allocate(bytes); });
  }

  char* AllocateAligned(size_t bytes, size_t huge_page_size = 0,
                        Logger* logger = nullptr) override {
    size_t rounded_up = ((bytes - 1) | (sizeof(void*) - 1)) + 1;
    assert(rounded_up >= bytes && rounded_up < bytes + sizeof(void*) &&
           (rounded_up % sizeof(void*)) == 0);

    return AllocateImpl(rounded_up, huge_page_size != 0 /*force_arena*/,
                        [this, rounded_up, huge_page_size, logger]() {
                          return arena_.AllocateAligned(rounded_up,
                                                        huge_page_size, logger);
                        });
  }

  size_t ApproximateMemoryUsage() const {
    std::unique_lock<SpinMutex> lock(arena_mutex_, std::defer_lock);
    lock.lock();
    return arena_.ApproximateMemoryUsage() - ShardAllocatedAndUnused();
  }

  size_t MemoryAllocatedBytes() const {
    return memory_allocated_bytes_.load(std::memory_order_relaxed);
  }

  size_t AllocatedAndUnused() const {
    return arena_allocated_and_unused_.load(std::memory_order_relaxed) +
           ShardAllocatedAndUnused();
  }

  size_t IrregularBlockNum() const {
    return irregular_block_num_.load(std::memory_order_relaxed);
  }

  size_t BlockSize() const override { return arena_.BlockSize(); }

 private:
  struct Shard {
    char padding[40] ROCKSDB_FIELD_UNUSED;
    mutable SpinMutex mutex;
    char* free_begin_;
    std::atomic<size_t> allocated_and_unused_;

    Shard() : free_begin_(nullptr), allocated_and_unused_(0) {}
  };

  static thread_local size_t tls_cpuid;

  char padding0[56] ROCKSDB_FIELD_UNUSED;

  size_t shard_block_size_;

  CoreLocalArray<Shard> shards_;

  Arena arena_;
  mutable SpinMutex arena_mutex_;
  std::atomic<size_t> arena_allocated_and_unused_;
  std::atomic<size_t> memory_allocated_bytes_;
  std::atomic<size_t> irregular_block_num_;

  char padding1[56] ROCKSDB_FIELD_UNUSED;

  Shard* Repick();

  size_t ShardAllocatedAndUnused() const {
    size_t total = 0;
    for (size_t i = 0; i < shards_.Size(); ++i) {
      total += shards_.AccessAtCore(i)->allocated_and_unused_.load(
          std::memory_order_relaxed);
    }
    return total;
  }

  /**
   * 分配策略决策：
   *
   * 如果分配大小超过分片块大小的1/4，或者强制使用arena，或者当前线程尚未初始化分片缓存，则直接使用主arena分配
   * 否则使用每核心分片缓存进行分配

   * 线程安全处理：
   * 使用自旋锁（SpinMutex）保护关键区域
   * 采用try_lock机制避免线程阻塞
   * 如果当前分片被占用，通过Repick()选择其他可用分片

   * 内存管理优化：
   * 每核心分片维护自己的空闲内存指针和已分配统计
   * 当分片内存不足时，从主arena申请新的内存块
   * 支持对齐和非对齐分配

   * 性能优化：
   * 延迟初始化分片，避免不必要的内存浪费
   * 动态调整分片块大小以适应实际使用模式
   * 使用Fixup()方法同步主arena和分片的统计信息
   */

  template <typename Func>
  char* AllocateImpl(size_t bytes, bool force_arena, const Func& func) {
    size_t cpu;

    // Go directly to the arena if the allocation is too large, or if
    // we've never needed to Repick() and the arena mutex is available
    // with no waiting.  This keeps the fragmentation penalty of
    // concurrency zero unless it might actually confer an advantage.
    std::unique_lock<SpinMutex> arena_lock(arena_mutex_, std::defer_lock);
    if (bytes > shard_block_size_ / 4 || force_arena ||
        ((cpu = tls_cpuid) == 0 &&
         !shards_.AccessAtCore(0)->allocated_and_unused_.load(
             std::memory_order_relaxed) &&
         arena_lock.try_lock())) {
      if (!arena_lock.owns_lock()) {
        arena_lock.lock();
      }
      auto rv = func();
      Fixup();
      return rv;
    }

    // pick a shard from which to allocate
    Shard* s = shards_.AccessAtCore(cpu & (shards_.Size() - 1));
    if (!s->mutex.try_lock()) {
      s = Repick();
      s->mutex.lock();
    }
    std::unique_lock<SpinMutex> lock(s->mutex, std::adopt_lock);

    size_t avail = s->allocated_and_unused_.load(std::memory_order_relaxed);
    if (avail < bytes) {
      // reload
      std::lock_guard<SpinMutex> reload_lock(arena_mutex_);

      // If the arena's current block is within a factor of 2 of the right
      // size, we adjust our request to avoid arena waste.
      auto exact = arena_allocated_and_unused_.load(std::memory_order_relaxed);
      assert(exact == arena_.AllocatedAndUnused());

      if (exact >= bytes && arena_.IsInInlineBlock()) {
        // If we haven't exhausted arena's inline block yet, allocate from arena
        // directly. This ensures that we'll do the first few small allocations
        // without allocating any blocks.
        // In particular this prevents empty memtables from using
        // disproportionately large amount of memory: a memtable allocates on
        // the order of 1 KB of memory when created; we wouldn't want to
        // allocate a full arena block (typically a few megabytes) for that,
        // especially if there are thousands of empty memtables.
        auto rv = func();
        Fixup();
        return rv;
      }

      avail = exact >= shard_block_size_ / 2 && exact < shard_block_size_ * 2
                  ? exact
                  : shard_block_size_;
      s->free_begin_ = arena_.AllocateAligned(avail);
      Fixup();
    }
    s->allocated_and_unused_.store(avail - bytes, std::memory_order_relaxed);

    char* rv;
    if ((bytes % sizeof(void*)) == 0) {
      // aligned allocation from the beginning
      rv = s->free_begin_;
      s->free_begin_ += bytes;
    } else {
      // unaligned from the end
      rv = s->free_begin_ + avail - bytes;
    }
    return rv;
  }

  void Fixup() {
    arena_allocated_and_unused_.store(arena_.AllocatedAndUnused(),
                                      std::memory_order_relaxed);
    memory_allocated_bytes_.store(arena_.MemoryAllocatedBytes(),
                                  std::memory_order_relaxed);
    irregular_block_num_.store(arena_.IrregularBlockNum(),
                               std::memory_order_relaxed);
  }

  ConcurrentArena(const ConcurrentArena&) = delete;
  ConcurrentArena& operator=(const ConcurrentArena&) = delete;
};

}  // namespace ROCKSDB_NAMESPACE
