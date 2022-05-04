#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

typedef struct {
    int* output;
    int threadId;
} WorkerArgs;


extern void mandelbrotSerial(
    int startRow, int numRows, int output[]);

extern float x0, x1, y0, y1;
extern unsigned int width, height;
extern int maxIterations, threadId, numThreads;
//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs * const args) {
    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    int counter = 1200 / numThreads + 1;
    for(int i = 0; i < counter; ++i){
        if(2 * (i * numThreads + args->threadId) >= 1200){
            break;
        }
        mandelbrotSerial(2 * (i * numThreads + args->threadId), 2, args->output);
    }
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread`
        args[i].output = output;
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}
