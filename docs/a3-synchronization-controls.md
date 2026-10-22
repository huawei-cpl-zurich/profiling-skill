# A3 synchronization placement controls

Run `python benchmarks/a3_profile_sync.py --manifest` without device dependencies.
On an approved A3 named runtime, first use `--correctness-only --output NEW_DIR`
and select one or more `--case` IDs. Omit `--correctness-only` to collect five
BasicInfo and three PipeUtilization replays with the existing batch runner.
The runner acquires one idle device lease for the batch and applications use
logical device zero. Submit through the global remote-access client; never
invoke these scripts over an alternate transport.
The batch stops on its first failed correctness/acquisition cell; inspect its
retained command and diagnostics before scheduling any remaining variants.

All variants preserve useful loads, vector work, and outputs. Intra-core
controls compare early signal/late wait (`overlap`), early wait, and late
signal. The MTE2→Vector late signal includes the second useful input transfer;
the Vector→MTE3 late signal includes independent vector work. Every event is
paired, and independent vector work is stored so it cannot be eliminated.

Mode 0 uses two vector participants, batch scheduling, one all-core barrier,
and reads the other participant's published output after that barrier. Mode 2
uses an AIC/AIV handshake: both vector subcores publish output and signal,
then receive the Cube acknowledgement. Its AIC side overlaps an independent
GM→L1 copy with the wait. Numerical correctness validates vector output and
handshake completion, not the contents of that independent AIC copy. Neither
mode uses high-level Matmul APIs or more than one outstanding cross-core flag
increment. Balanced and imbalanced variants retain different expected outputs;
compare wait placement only within the same balance group.

`a3_native_build.build(output, source=None, compiler=None, extra_flags=())`
also supports standalone sanitizer/control sources. It retains the exact
compiler command, source and executable hashes, and compiler diagnostics.
Instrumentation must have its own binary/cache and must never be timed as an
ordinary performance candidate.

Keep BasicInfo latency separate from per-core/subcore pipe times. Early versus
late waits can alter issue order without making an exported wait counter move.
Compiler/disassembly evidence must establish actual emitted placement before
attributing a metric change. Missing metrics remain unavailable, not zero;
independent replays cannot establish simultaneous overlap.

The installed CANN 9.1 HIIPU objdump may exit zero while every instruction is
`<not available>`. The builder records this as unavailable, retaining the
objects and preprocessed `.asci` rather than calling that decoded assembly.
Such runs compare source-controlled placement variants; instruction-level
placement and absence of extra compiler synchronization remain unverified.

Primary API reference: CANN Ascend C `SetFlag/WaitFlag`,
`CrossCoreSetFlag/CrossCoreWaitFlag` (A3 modes 0 and 2), and the
[direct compilation guide](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.html).
