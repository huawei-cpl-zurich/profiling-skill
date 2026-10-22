#include <cassert>
#include <cstdint>
#define __aicore__
#define __global__
#define __gm__
#define __schedmode__(...)
#define GM_ADDR unsigned char*
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define ASCEND_IS_AIC (cpu_aic)
#define ASCEND_IS_AIV (!cpu_aic)
inline bool cpu_aic = false;
inline int cpu_block = 0;
namespace AscendC {
enum class HardEvent { MTE2_V, V_MTE3 };
enum class TPosition { VECCALC, A1 };
enum Pipe { PIPE_V, PIPE_ALL, PIPE_MTE3, PIPE_MTE2 };
using TEventID = int8_t;
inline int allocations = 0, releases = 0, peak[2] = {}, cross_calls = 0;
struct TPipe;
inline TPipe* active;
struct TPipe {
    bool owned[2][8] = {}, armed[2][8] = {};
    TPipe() { assert(!active); active = this; }
    ~TPipe() {
        for (int e = 0; e < 2; ++e) for (int i = 0; i < 8; ++i)
            assert(!owned[e][i] && !armed[e][i]);
        active = nullptr;
    }
    template<class T> void InitBuffer(T&, int) {}
    template<HardEvent e> TEventID AllocEventID() {
        const int event = int(e);
        for (int i = 2; i < 8; ++i) if (!owned[event][i]) {
            owned[event][i] = true;
            int live = 0;
            for (bool value : owned[event]) live += value;
            if (live > peak[event]) peak[event] = live;
            ++allocations;
            return i;
        }
        assert(false); return -1;
    }
    template<HardEvent e> void ReleaseEventID(TEventID id) {
        assert(id >= 2 && id < 8 && owned[int(e)][id] && !armed[int(e)][id]);
        owned[int(e)][id] = false; ++releases;
    }
};
template<HardEvent e> void SetFlag(TEventID id) {
    assert(active && id >= 2 && id < 8 && active->owned[int(e)][id]);
    assert(!active->armed[int(e)][id]); active->armed[int(e)][id] = true;
}
template<HardEvent e> void WaitFlag(TEventID id) {
    assert(active && id >= 2 && id < 8 && active->owned[int(e)][id]);
    assert(active->armed[int(e)][id]); active->armed[int(e)][id] = false;
}
template<class T> struct LocalTensor { LocalTensor operator[](int) const { return {}; } };
template<class T> struct GlobalTensor {
    void SetGlobalBuffer(T*, int) {}
    GlobalTensor operator[](int) const { return {}; }
};
template<TPosition p> struct TBuf { template<class T> LocalTensor<T> Get() { return {}; } };
template<class... T> void DataCopy(T...) {}
template<class... T> void Duplicate(T...) {}
template<class... T> void Adds(T...) {}
template<class... T> void Add(T...) {}
template<Pipe p> void PipeBarrier() {}
inline int GetBlockIdx() { return cpu_block; }
inline int GetSubBlockIdx() { return cpu_block % 2; }
template<int mode, Pipe p> void CrossCoreSetFlag(int id) { assert(id == 8 || id == 9); ++cross_calls; }
template<int mode = 0> void CrossCoreWaitFlag(int id) { assert(id == 8 || id == 9); ++cross_calls; }
}
