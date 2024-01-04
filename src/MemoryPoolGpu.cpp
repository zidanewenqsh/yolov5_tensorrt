// MemoryPoolGpu.cpp
#include "MemoryPoolGpu.h"
// 检查 CUDA 运行时错误的辅助函数
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
static bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        std::cerr << "runtime error " << file << ":" << line << " " << op << " failed.\n"
                  << "  code = " << err_name << ", message = " << err_message << std::endl;   
        return false;
    }
    return true;
}


MemPoolGpu::MemPoolGpu(size_t blockSize) 
    : blockSize(blockSize), stop(false) {
    int blockCount = 10;
    for (int i = 0; i < blockCount; ++i) {
        char* block;
        checkRuntime(cudaMalloc((void**)&block, blockSize));
        freeBlocks.push_back(block);
    }
}

MemPoolGpu::MemPoolGpu(size_t blockSize, size_t blockCount) 
    : blockSize(blockSize), stop(false) {
    for (size_t i = 0; i < blockCount; ++i) {
        char* block;
        checkRuntime(cudaMalloc((void**)&block, blockSize));
        freeBlocks.push_back(block);
    }
}

MemPoolGpu::~MemPoolGpu() {
    for (char* block : freeBlocks) {
        checkRuntime(cudaFree(block));
    }
}

void* MemPoolGpu::allocate() {
    if (stop.load()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex);

    if (freeBlocks.empty()) {
        char* block;
        checkRuntime(cudaMalloc((void**)&block, blockSize));
        return block; 
    }

    char* block = freeBlocks.back();
    freeBlocks.pop_back();
    return block;
}

void MemPoolGpu::deallocate(void* block) {
    if (stop.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    freeBlocks.push_back(static_cast<char*>(block));
}

void MemPoolGpu::stopPool() {
    stop.store(true);
}





MemoryPoolGpu::MemoryPoolGpu():stop(false) {
    smallSize = 0x100;
    smallPoolSize = smallSize * 0x10;
    addSmallBlock(smallPoolSize);
    int i = 1, small_size = smallSize;
    while (small_size >>=1) i++;
    // printf("i=%d\n", i);
    size_t blockSize = 1 << i;
    largeBlockSizes.push_back(blockSize);
    largePools.push_back({{}, blockSize}); // 为每个大小创建一个空内存池
}

MemoryPoolGpu::MemoryPoolGpu(size_t small_size, int small_count)
    : stop(false), smallSize(small_size), smallPoolSize(smallSize * small_count) {
    // ... 构造函数的实现 ...
    // 初始化小块内存
    addSmallBlock(smallPoolSize);

    // 定义大块内存的潜在大小
    int i = 1;
    while (small_size >>=1) i++;
    // printf("i=%d\n", i);
    size_t blockSize = 1 << i;
    largeBlockSizes.push_back(blockSize);
    largePools.push_back({{}, blockSize}); // 为每个大小创建一个空内存池
}

MemoryPoolGpu::~MemoryPoolGpu() {
    // ... 析构函数的实现 ...
    // 释放所有小块内存的大块
    for (auto& block : smallPools) {
        checkRuntime(cudaFree(block.memory));
    }
    // 释放所有大块内存
    for (auto& pool : largePools) {
        for (auto& block : pool.blocks) {
            if (block.memory != nullptr) {
                checkRuntime(cudaFree(block.memory));
            }
        }
    }
}

void* MemoryPoolGpu::allocate(size_t size) {
    // ... allocate 的实现 ...
    if (stop.load()) {
        return nullptr; // 如果内存池已停止，直接返回
    }
    if (size <= smallSize) {
        // 在小块内存中分配
        return allocateSmall(size);
    } else {
        // 在大块内存中分配
        return allocateLarge(size);
    }
}

void MemoryPoolGpu::deallocate(void* block) {
    // ... deallocate 的实现 ...
    if (stop.load()) {
        return ; // 如果内存池已停止，直接返回
    }
    while (spinlock.test_and_set(std::memory_order_acquire)); // 获取锁
    // 检查是否为大块内存
    for (auto& pool : largePools) {
        for (auto& blockInPool : pool.blocks) {
            if (blockInPool.memory == block) {
                blockInPool.inUse = false; // 标记为未使用
                std::cout << "deallocate " << block << std::endl; 
                spinlock.clear(std::memory_order_release); // 释放锁
                return;
            }
        }
    }
    spinlock.clear(std::memory_order_release); // 释放锁
}

void MemoryPoolGpu::stopPool() {
    // ... stopPool 的实现 ...
    // stop.store(1); // 设置停止标志
    stop.store(true);
}

// ... 其他成员函数的实现 ...

void MemoryPoolGpu::addSmallBlock(size_t size) {
    // ... addSmallBlock 的实现 ...
    SmallMemoryBlock newBlock;
    newBlock.totalSize = size;
    newBlock.freeSize = size;
    checkRuntime(cudaMalloc(&newBlock.memory, size));
    newBlock.nextAvailable = static_cast<char*>(newBlock.memory);
    smallPools.push_back(newBlock);
}

void* MemoryPoolGpu::allocateSmall(size_t size) {
    // ... allocateSmall 的实现 ...
    while (spinlock.test_and_set(std::memory_order_acquire)); // 获取锁

    // 尝试在现有的小块内存中分配
    for (auto& block : smallPools) {
        if (block.freeSize >= size) {
            void* allocatedMemory = block.nextAvailable;
            block.nextAvailable += size;
            block.freeSize -= size;
            spinlock.clear(std::memory_order_release); // 释放锁
            return allocatedMemory;
        }
    }

    // 所有现有的小块内存都已满，分配一个新的大块
    addSmallBlock(smallPools[0].totalSize);
    SmallMemoryBlock& newBlock = smallPools.back(); // 获取刚刚添加的新块

    // 直接在新块中分配内存
    void* allocatedMemory = newBlock.nextAvailable;
    newBlock.nextAvailable += size;
    newBlock.freeSize -= size;

    spinlock.clear(std::memory_order_release); // 释放锁
    return allocatedMemory;
}

void* MemoryPoolGpu::allocateLarge(size_t size) {
    // ... allocateLarge 的实现 ...
    while (spinlock.test_and_set(std::memory_order_acquire)); // 获取锁
    // 如果预设的largeBlockSize不满足要求，扩容
    while (size > largeBlockSizes.back()) {
        largeBlockSizes.push_back(largeBlockSizes.back() * 2);
        largePools.push_back({{}, largeBlockSizes.back()}); // 为每个大小创建一个空内存池
    }
    for (auto& pool : largePools) {
        if (pool.blockSize >= size) {
            for (auto& block : pool.blocks) {
                if (!block.inUse) {
                    block.inUse = true;
                    spinlock.clear(std::memory_order_release); // 释放锁
                    return block.memory;
                }
            }

            LargeMemoryBlock newBlock;
            checkRuntime(cudaMalloc(&newBlock.memory, pool.blockSize));
            newBlock.inUse = true;
            pool.blocks.push_back(newBlock);
            spinlock.clear(std::memory_order_release); // 释放锁
            return newBlock.memory;
        }
    }
    spinlock.clear(std::memory_order_release); // 释放锁
    throw std::bad_alloc();
}