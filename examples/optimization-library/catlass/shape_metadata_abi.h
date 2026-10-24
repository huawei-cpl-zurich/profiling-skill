// Portable host-only configuration shared by the two independent loader images.
#pragma once
#include <cstdint>
namespace ShapeMetadata {
struct State {
    uint32_t m=0,n=0,k=0,cores=0;
    int configure(uint32_t rows,uint32_t columns,uint32_t reduction,uint32_t cube_cores) {
        if (!rows || !columns || !reduction || !cube_cores || rows%16 ||
            columns%16 || reduction%16 || reduction>256 ||
            ((uint64_t(rows)+127)/128)*((uint64_t(columns)+255)/256)>cube_cores) return 1;
        m=rows; n=columns; k=reduction; cores=cube_cores; return 0;
    }
    int validate(void** pointers,uint64_t count) const {
        return count!=3 || !cores || !pointers || !pointers[0] ||
            !pointers[1] || !pointers[2] ? 2 : 0;
    }
};
}
