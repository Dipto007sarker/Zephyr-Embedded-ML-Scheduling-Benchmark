#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>
#include <stdint.h>

#include "main_functions.hpp"

#define ML_STACK_SIZE 4096
#define RT_STACK_SIZE 2048
#define BG_STACK_SIZE 2048

#define ML_PRIO 4
#define RT_PRIO 4
#define BG_PRIO 6

K_THREAD_STACK_DEFINE(ml_stack, ML_STACK_SIZE);
K_THREAD_STACK_DEFINE(rt_stack, RT_STACK_SIZE);
K_THREAD_STACK_DEFINE(bg_stack, BG_STACK_SIZE);

static struct k_thread ml_thread_data;
static struct k_thread rt_thread_data;
static struct k_thread bg_thread_data;

volatile uint32_t ml_counter = 0;
volatile uint32_t rt_misses = 0;
volatile uint32_t rt_runs = 0;

void ml_task(void *a, void *b, void *c)
{
    ARG_UNUSED(a);
    ARG_UNUSED(b);
    ARG_UNUSED(c);

    setup();

    const int inference_repeats = 1;   // start with 1 for Magic Wand

    while (1) {
        timing_t start = timing_counter_get();

        for (int i = 0; i < inference_repeats; i++) {
            loop();
        }

        timing_t end = timing_counter_get();
        uint64_t total_cycles = timing_cycles_get(&start, &end);
        uint64_t avg_cycles = total_cycles / inference_repeats;

        ml_counter++;
        printk("[ML] run=%u repeats=%d total_cycles=%llu avg_cycles=%llu\n",
               ml_counter, inference_repeats, total_cycles, avg_cycles);

        k_sleep(K_MSEC(20));
    }
}

void rt_task(void *a, void *b, void *c)
{
    ARG_UNUSED(a);
    ARG_UNUSED(b);
    ARG_UNUSED(c);

    const int period_ms = 200;
    int64_t next_release = k_uptime_get() + period_ms;
    int64_t last_start = 0;

    while (1) {
        int64_t now = k_uptime_get();
        int64_t lateness = now - next_release;
        int64_t gap = (last_start == 0) ? 0 : (now - last_start);

        if (lateness < 0) {
            lateness = 0;
        }

        if (lateness > period_ms) {
            rt_misses++;
        }

        int64_t start = k_uptime_get();
        last_start = start;

        k_busy_wait(5000);

        int64_t finish = k_uptime_get();
        rt_runs++;

        printk("[RT] run=%u exec_ms=%lld lateness_ms=%lld gap_ms=%lld misses=%u\n",
               rt_runs, (finish - start), lateness, gap, rt_misses);

        next_release += period_ms;

        while (next_release <= k_uptime_get()) {
            next_release += period_ms;
        }

        int64_t sleep_ms = next_release - k_uptime_get();
        if (sleep_ms > 0) {
            k_sleep(K_MSEC(sleep_ms));
        }
    }
}

void bg_task(void *a, void *b, void *c)
{
    ARG_UNUSED(a);
    ARG_UNUSED(b);
    ARG_UNUSED(c);

    while (1) {
        for (volatile uint32_t i = 0; i < 200000; i++) {
        }
        k_yield();
    }
}

int main(void)
{
    timing_init();
    timing_start();

    printk("Zephyr Magic Wand scheduling benchmark start\n");

    k_tid_t ml_tid = k_thread_create(&ml_thread_data, ml_stack, ML_STACK_SIZE,
                                     ml_task, NULL, NULL, NULL,
                                     ML_PRIO, 0, K_NO_WAIT);

    k_tid_t rt_tid = k_thread_create(&rt_thread_data, rt_stack, RT_STACK_SIZE,
                                     rt_task, NULL, NULL, NULL,
                                     RT_PRIO, 0, K_NO_WAIT);

    k_tid_t bg_tid = k_thread_create(&bg_thread_data, bg_stack, BG_STACK_SIZE,
                                     bg_task, NULL, NULL, NULL,
                                     BG_PRIO, 0, K_NO_WAIT);

    k_thread_name_set(ml_tid, "ml_task");
    k_thread_name_set(rt_tid, "rt_task");
    k_thread_name_set(bg_tid, "bg_task");

    // Uncomment only for equal-priority with deadline experiment
    k_thread_priority_set(ml_tid, 4);
    k_thread_priority_set(rt_tid, 4);
    k_thread_deadline_set(ml_tid, 30);
    k_thread_deadline_set(rt_tid, 10);

    while (1) {
        k_sleep(K_SECONDS(5));
        printk("[SUMMARY] ml_runs=%u rt_runs=%u rt_misses=%u\n",
               ml_counter, rt_runs, rt_misses);
    }

    return 0;
}