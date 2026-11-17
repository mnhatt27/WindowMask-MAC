/* Universal native launcher and upgrade stopper. No installed Python needed. */
#include <errno.h>
#include <libproc.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

static int is_backend(pid_t pid) {
    char path[PROC_PIDPATHINFO_MAXSIZE];
    if (proc_pidpath(pid, path, sizeof(path)) <= 0) return 0;
    return !strcmp(path, "/Applications/WindowMask Backend.app/Contents/Resources/runtimes/arm64/WindowMaskRuntime/WindowMaskRuntime") ||
           !strcmp(path, "/Applications/WindowMask Backend.app/Contents/Resources/runtimes/x86_64/WindowMaskRuntime/WindowMaskRuntime");
}

static int stop_backends(void) {
    int count = proc_listallpids(NULL, 0) + 256;
    pid_t *pids = calloc((size_t)count, sizeof(pid_t));
    if (!pids) return 1;
    count = proc_listallpids(pids, count * (int)sizeof(pid_t));
    if (count < 0) { free(pids); return 1; }
    int failed = 0;
    for (int i = 0; i < count; ++i) {
        if (pids[i] <= 1 || !is_backend(pids[i])) continue;
        if (kill(pids[i], SIGTERM) && errno != ESRCH) { failed = 1; continue; }
        int attempts = 100;
        while (attempts-- && is_backend(pids[i])) usleep(100000);
        if (is_backend(pids[i])) {
            if (kill(pids[i], SIGKILL) && errno != ESRCH) failed = 1;
            attempts = 50;
            while (attempts-- && is_backend(pids[i])) usleep(100000);
            if (is_backend(pids[i])) failed = 1;
        }
    }
    free(pids);
    return failed;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--stop")) return stop_backends();
    char raw[PATH_MAX], executable[PATH_MAX], runtime[PATH_MAX];
    uint32_t capacity = sizeof(raw);
    if (_NSGetExecutablePath(raw, &capacity) || !realpath(raw, executable)) return 1;
    char *slash = strrchr(executable, '/');
    if (!slash) return 1;
    *slash = '\0';
    int arm = 0;
    size_t length = sizeof(arm);
    (void)sysctlbyname("hw.optional.arm64", &arm, &length, NULL, 0);
    int needed = snprintf(runtime, sizeof(runtime), "%s/../Resources/runtimes/%s/WindowMaskRuntime/WindowMaskRuntime", executable, arm ? "arm64" : "x86_64");
    if (needed < 0 || (size_t)needed >= sizeof(runtime)) return 1;
    char **arguments = calloc((size_t)argc + 1, sizeof(char *));
    if (!arguments) return 1;
    arguments[0] = runtime;
    for (int i = 1; i < argc; ++i) arguments[i] = argv[i];
    execv(runtime, arguments);
    perror("WindowMask runtime");
    free(arguments);
    return 1;
}
