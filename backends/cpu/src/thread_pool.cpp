#include <tc/cpu/cpu_device.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace tc::cpu {

namespace {

// Persistent workers taking tasks from the jobs of concurrent bulk() calls.
// The calling thread works on its own job too, so a bulk() issued from a
// task of another bulk() always completes.
struct ThreadPool final : ParallelExecutor {
    explicit ThreadPool(std::uint32_t const workers) {
        threads.reserve(workers);
        for (std::uint32_t i = 0; i < workers; ++i) {
            threads.emplace_back([this] { work(); });
        }
    }

    ThreadPool(ThreadPool const &) = delete;
    ThreadPool &operator=(ThreadPool const &) = delete;

    ~ThreadPool() override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            stopping = true;
        }
        wake.notify_all();
        for (auto &thread : threads) {
            thread.join();
        }
    }

    std::uint32_t concurrency() const override { return static_cast<std::uint32_t>(threads.size()) + 1; }

    void bulk(std::uint32_t const count, TaskRef const task) override {
        if (count == 0) {
            return;
        }
        if (count == 1 or threads.empty()) {
            for (std::uint32_t i = 0; i < count; ++i) {
                task(i);
            }
            return;
        }
        auto job = Job{task, count};
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            jobs.push_back(&job);
        }
        wake.notify_all();
        run(job);
        auto lock = std::unique_lock<std::mutex>{mutex};
        retire(job);
        finished.wait(lock, [&] { return job.done.load() == count and job.users == 0; });
    }

private:
    struct Job {
        Job(TaskRef const task_, std::uint32_t const count_) : task(task_), count(count_) {}

        TaskRef task;
        std::uint32_t count;
        std::atomic<std::uint32_t> next{0};
        std::atomic<std::uint32_t> done{0};
        std::uint32_t users = 0; // workers inside run(); guarded by the pool mutex
    };

    void run(Job &job) {
        for (auto i = job.next.fetch_add(1); i < job.count; i = job.next.fetch_add(1)) {
            job.task(i);
            if (job.done.fetch_add(1) + 1 == job.count) {
                auto const lock = std::lock_guard<std::mutex>{mutex};
                finished.notify_all();
            }
        }
    }

    // Takes an exhausted job out of the queue; needs the mutex.
    void retire(Job &job) {
        auto const it = std::find(jobs.begin(), jobs.end(), &job);
        if (it != jobs.end()) {
            jobs.erase(it);
        }
    }

    void work() {
        auto lock = std::unique_lock<std::mutex>{mutex};
        for (;;) {
            wake.wait(lock, [&] { return stopping or not jobs.empty(); });
            if (stopping) {
                return;
            }
            auto &job = *jobs.front();
            if (job.next.load() >= job.count) {
                retire(job);
                continue;
            }
            ++job.users;
            lock.unlock();
            run(job);
            lock.lock();
            --job.users;
            retire(job);
            finished.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable finished;
    std::deque<Job *> jobs;
    bool stopping = false;
    std::vector<std::thread> threads;
};

} // namespace

std::shared_ptr<ParallelExecutor> makeThreadPool(std::uint32_t const threads) {
    auto const total = threads != 0 ? threads : std::max(1u, std::thread::hardware_concurrency());
    return std::make_shared<ThreadPool>(total - 1);
}

} // namespace tc::cpu
