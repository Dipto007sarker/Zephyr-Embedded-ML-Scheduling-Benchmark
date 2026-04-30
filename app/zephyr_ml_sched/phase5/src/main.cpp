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
volatile uint32_t ml_attempt_counter = 0;
volatile uint32_t rt_misses = 0;
volatile uint32_t rt_runs = 0;

/* Controller states */
enum AdaptiveMode : uint32_t {
    MODE_NORMAL = 0,
    MODE_RECOVERY = 1,
    MODE_PROTECT = 2,
};

volatile uint32_t adaptive_mode = MODE_NORMAL;
volatile uint32_t healthy_streak = 0;
volatile uint32_t last_rt_misses_seen = 0;
volatile int64_t recovery_until_ms = 0;

/* Thresholds */
static constexpr int64_t ENTER_PROTECTION_LATENESS_MS = 50;
static constexpr int64_t EXIT_PROTECTION_LATENESS_MS = 20;
static constexpr uint32_t HEALTHY_STREAK_TO_EXIT = 20;

/* Recovery window */
static constexpr int RECOVERY_WINDOW_MS = 10000;

/* Sleep policy */
static constexpr int NORMAL_ML_SLEEP_MS = 20;
static constexpr int PROTECT_ML_SLEEP_MS = 500;
static constexpr int RECOVERY_ML_SLEEP_MS = 20;

/* Protection throttling */
static constexpr uint32_t PROTECT_SKIP_FACTOR = 50;

static const char *mode_to_str(uint32_t mode)
{
    switch (mode) {
    case MODE_NORMAL:
        return "NORMAL";
    case MODE_RECOVERY:
        return "RECOVERY";
    case MODE_PROTECT:
        return "PROTECT";
    default:
        return "UNKNOWN";
    }
}

void ml_task(void *a, void *b, void *c)
{
    ARG_UNUSED(a);
    ARG_UNUSED(b);
    ARG_UNUSED(c);

    setup();

    const int inference_repeats = 1;
    uint32_t protect_iteration = 0;

    while (1) {
        ml_attempt_counter++;

        uint32_t mode_snapshot = adaptive_mode;
        bool should_run_ml = true;

        if (mode_snapshot == MODE_RECOVERY) {
            should_run_ml = false;
            protect_iteration = 0;
        } else if (mode_snapshot == MODE_PROTECT) {
            protect_iteration++;
            should_run_ml = ((protect_iteration % PROTECT_SKIP_FACTOR) == 1);
        } else {
            protect_iteration = 0;
            should_run_ml = true;
        }

        uint64_t total_cycles = 0;
        uint64_t avg_cycles = 0;

        if (should_run_ml) {
            timing_t start = timing_counter_get();

            for (int i = 0; i < inference_repeats; i++) {
                loop();
            }

            timing_t end = timing_counter_get();
            total_cycles = timing_cycles_get(&start, &end);
            avg_cycles = total_cycles / inference_repeats;

            ml_counter++;

            printk("[ML] attempt=%u run=%u repeats=%d total_cycles=%llu avg_cycles=%llu mode=%s action=RUN\n",
                   ml_attempt_counter,
                   ml_counter,
                   inference_repeats,
                   total_cycles,
                   avg_cycles,
                   mode_to_str(mode_snapshot));
        } else {
            printk("[ML] attempt=%u run=%u repeats=%d total_cycles=0 avg_cycles=0 mode=%s action=BLOCK\n",
                   ml_attempt_counter,
                   ml_counter,
                   inference_repeats,
                   mode_to_str(mode_snapshot));
        }

        if (mode_snapshot == MODE_NORMAL) {
            k_sleep(K_MSEC(NORMAL_ML_SLEEP_MS));
        } else if (mode_snapshot == MODE_RECOVERY) {
            k_sleep(K_MSEC(RECOVERY_ML_SLEEP_MS));
        } else {
            k_sleep(K_MSEC(PROTECT_ML_SLEEP_MS));
        }
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

        bool new_miss = (rt_misses > last_rt_misses_seen);
        bool unhealthy = (lateness > ENTER_PROTECTION_LATENESS_MS) || new_miss;
        bool healthy = (lateness <= EXIT_PROTECTION_LATENESS_MS) && !new_miss;

        int64_t now_after_exec = k_uptime_get();

        if (unhealthy) {
            adaptive_mode = MODE_RECOVERY;
            recovery_until_ms = now_after_exec + RECOVERY_WINDOW_MS;
            healthy_streak = 0;
        } else {
            if (adaptive_mode == MODE_RECOVERY) {
                if (now_after_exec >= recovery_until_ms) {
                    adaptive_mode = MODE_PROTECT;
                }
            } else if (adaptive_mode == MODE_PROTECT) {
                if (healthy) {
                    healthy_streak++;
                    if (healthy_streak >= HEALTHY_STREAK_TO_EXIT) {
                        adaptive_mode = MODE_NORMAL;
                    }
                } else {
                    healthy_streak = 0;
                }
            } else { /* MODE_NORMAL */
                if (healthy) {
                    healthy_streak++;
                } else {
                    healthy_streak = 0;
                }
            }
        }

        last_rt_misses_seen = rt_misses;

        printk("[RT] run=%u exec_ms=%lld lateness_ms=%lld gap_ms=%lld misses=%u mode=%s healthy_streak=%u recovery_until=%lld\n",
               rt_runs,
               (finish - start),
               lateness,
               gap,
               rt_misses,
               mode_to_str(adaptive_mode),
               healthy_streak,
               recovery_until_ms);

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

    printk("Zephyr Magic Wand adaptive scheduling benchmark start\n");

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

    /* equal priority + deadline */
    k_thread_priority_set(ml_tid, 4);
    k_thread_priority_set(rt_tid, 4);
    k_thread_deadline_set(ml_tid, 30);
    k_thread_deadline_set(rt_tid, 10);

    while (1) {
        k_sleep(K_SECONDS(5));
        printk("[SUMMARY] ml_attempts=%u ml_runs=%u rt_runs=%u rt_misses=%u mode=%s healthy_streak=%u recovery_until=%lld\n",
               ml_attempt_counter,
               ml_counter,
               rt_runs,
               rt_misses,
               mode_to_str(adaptive_mode),
               healthy_streak,
               recovery_until_ms);
    }

    return 0;
}