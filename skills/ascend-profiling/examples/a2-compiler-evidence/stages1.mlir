// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @paired_matmul(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c2_i64 = arith.constant 2 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64_0 = arith.constant 1 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c20_i32 = arith.constant 20 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c20_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg9, %arg8 : i32
    %6 = arith.remsi %5, %arg7 : i32
    %7 = arith.divsi %arg9, %4 : i32
    %8 = arith.remsi %7, %arg6 : i32
    %9 = arith.muli %8, %c32_i32 : i32
    %10 = arith.muli %6, %c32_i32 : i32
    %11 = arith.index_cast %9 : i32 to index
    %12 = affine.max affine_map<()[s0] -> (512, s0)>()[%11]
    %13 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%12, %11]
    %14 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%13, %11]
    %15 = affine.max affine_map<()[s0] -> (0, s0)>()[%14]
    %16 = arith.cmpi slt, %15, %c32 : index
    %17 = arith.index_cast %10 : i32 to index
    %18 = affine.max affine_map<()[s0] -> (512, s0)>()[%17]
    %19 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%18, %17]
    %20 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%19, %17]
    %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
    %22 = arith.cmpi slt, %21, %c32 : index
    %23 = hivm.hir.pointer_cast(%c0_i64_1) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %23 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg10 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %27 = arith.index_cast %arg10 : i32 to index
      %28 = arith.index_cast %c0_i32 : i32 to index
      %29 = arith.index_cast %c8_i32 : i32 to index
      %30 = arith.index_cast %c1_i32 : i32 to index
      %31 = arith.index_cast %arg9 : i32 to index
      %32 = arith.index_cast %3 : i32 to index
      %33 = arith.index_cast %1 : i32 to index
      %34 = arith.index_cast %c20_i32 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>()[%27, %28, %29, %30, %31, %32, %34]
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.muli %arg10, %c64_i32 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%38, %11]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [32, 64], strides: [512, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x64xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %40 = hivm.hir.pointer_cast(%c0_i64_1) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
      %cast_5 = memref.cast %40 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %41 = affine.max affine_map<()[s0] -> (512, s0)>()[%38]
      %42 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%41, %38]
      %43 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%42, %38]
      %44 = affine.max affine_map<()[s0] -> (0, s0)>()[%43]
      %45 = arith.cmpi slt, %44, %c64 : index
      %46 = arith.ori %16, %45 : i1
      %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%15, %44] [1, 1] : memref<32x64xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%15]
      %48 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%44]
      %subview_7 = memref.subview %40[0, 0, 0, 0] [%48, %47, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_8 = memref.cast %subview_7 : memref<?x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %46 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %40 [[0, 1, 2, 3]] : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<?x?xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%17, %38]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [64, 32], strides: [512, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x32xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %50 = hivm.hir.pointer_cast(%c4096_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
      %cast_10 = memref.cast %50 : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %51 = arith.ori %45, %22 : i1
      %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%44, %21] [1, 1] : memref<64x32xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
      %52 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%21]
      %subview_12 = memref.subview %50[0, 0, 0, 0] [%52, %48, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x16x16xf16, strided<[1024, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_13 = memref.cast %subview_12 : memref<?x?x16x16xf16, strided<[1024, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      scf.if %51 {
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %50 [[0, 1, 2, 3]] : memref<2x4x16x16xf16, #hivm.address_space<cbuf>> into memref<2048xf16, #hivm.address_space<cbuf>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<2048xf16, #hivm.address_space<cbuf>>)
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.nd2nz {dst_continuous} ins(%subview_11 : memref<?x?xf16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %53 = arith.cmpi eq, %arg10, %c0_i32 : i32
      %c-1_i64 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_5, %cast_10, %53, %15, %44, %21 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64_0, %c1_i64, %c2_i64, %36, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %24 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 512)>()[%17, %11]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%24], sizes: [32, 32], strides: [512, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    %25 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%15]
    %26 = affine.apply affine_map<()[s0] -> ((s0 + 15) floordiv 16)>()[%21]
    %subview = memref.subview %23[0, 0, 0, 0] [%26, %25, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>>
    %cast_2 = memref.cast %subview : memref<?x?x16x16xf32, strided<[512, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
    %subview_3 = memref.subview %reinterpret_cast[0, 0] [%15, %21] [1, 1] : memref<32x32xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_2 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_3 : memref<?x?xf32, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
