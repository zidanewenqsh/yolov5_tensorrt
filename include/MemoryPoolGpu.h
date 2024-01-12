#ifndef __MEMORYPOOLGPU_H__
#define __MEMORYPOOLGPU_H__

#include <cstddef>
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>
#include <cuda_runtime.h>
#include <chrono>
#include <atomic>
#include "Logger.h"
// ... 可能还需要包含其他头文件 ...

class MemPoolGpu : public Logger {
public:
    MemPoolGpu(size_t blockSize);
    MemPoolGpu(size_t blockSize, size_t blockCount);
    ~MemPoolGpu();

    void* allocate();
    void deallocate(void* block);
    void stopPool();

private:
    size_t blockSize;
    std::vector<char*> freeBlocks;
    std::mutex mutex;
    std::atomic<bool> stop;
};

class MemoryPoolGpu {
public:
    MemoryPoolGpu();
    MemoryPoolGpu(size_t small_size, int small_count);
    ~MemoryPoolGpu();

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

    std::atomic<bool> stop;
    // std::atomic<int> stop;
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

#endif //__MEMORYPOOLGPU_H__