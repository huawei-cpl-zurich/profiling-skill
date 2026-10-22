// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @paired_memory(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c10240_i64 = arith.constant 10240 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c256_i32 = arith.constant 256 : i32
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c40_i32 = arith.constant 40 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c40_i32  : i32 {
    %5 = arith.index_cast %arg9 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c40_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c1_i64_1 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %13 = arith.select %12, %c1_i64_1, %c2_i64 : i64
    %14 = hivm.hir.pointer_cast(%c1024_i64, %c10240_i64) : memref<256x8xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<256x8xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c0_i64, %c9216_i64) : memref<256xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg9, %4 : i32
    %17 = arith.remsi %16, %arg6 : i32
    %18 = arith.muli %17, %c256_i32 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%19]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [256], strides: [2] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
    %21 = affine.max affine_map<()[s0] -> (1048576, s0)>()[%19]
    %22 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%21, %19]
    %23 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%22, %19]
    %24 = arith.cmpi slt, %23, %c256 : index
    %subview = memref.subview %reinterpret_cast[0] [%23] [1] : memref<256xf32, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %15[0] [%23] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %24 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%15 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
    } {hivm.unlikely_condition}
    %subview_3 = memref.subview %14[0, 0] [256, 1] [1, 1] : memref<256x8xf32, #hivm.address_space<ub>> to memref<256xf32, strided<[8]>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %subview_3[0] [%23] [1] : memref<256xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %13]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vdeinterleave ins(%subview_4 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%19], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%23] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %13]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
