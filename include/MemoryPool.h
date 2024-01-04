#ifndef __MEMORYPOOL_H__
#define __MEMORYPOOL_H__

#include <cstddef>
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>
#include <cuda_runtime.h>
#include <chrono>
#include <atomic>
// ... 可能还需要包含其他头文件 ...
class MemPool {
public:
    MemPool(size_t blockSize);
    MemPool(size_t blockSize, size_t blockCount);
    ~MemPool();

    void* allocate();
    void deallocate(void* block);
    void stopPool();

private:
    size_t blockSize;
    std::vector<char*> freeBlocks;
    std::mutex mutex;
    std::atomic<bool> stop;
};
#if 1
class MemoryPool {
public:
    MemoryPool();
    MemoryPool(size_t small_size, int small_count);
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* block);
    void stopPool();

private:
    struct LargeMemoryBlock {
        void* memory = nullptr;
        bool inUse = false;
    };

    struct LargePool {
        std::vector<LargeMemoryBlock> blocks;
        size_t blockSize;
    };

    struct SmallMemoryBlock {
        void* memory;
        char* nextAvailable;
        size_t freeSize;
        size_t totalSize;
    };

    // ... 其他私有成员变量和方法的声明 ...

    // std::atomic<int> stop;
    std::atomic<bool> stop;
    std::atomic_flag spinlock = ATOMIC_FLAG_INIT;
    size_t smallSize;
    size_t smallPoolSize;
    std::vector<LargePool> largePools;
    std::vector<size_t> largeBlockSizes;
    std::vector<SmallMemoryBlock> smallPools;

    void addSmallBlock(size_t size);
    void* allocateSmall(size_t size);
    void* allocateLarge(size_t size);
    // ... 其他私有成员函数的声明 ...
};
#endif
#endif // __MEMORYPOOL_