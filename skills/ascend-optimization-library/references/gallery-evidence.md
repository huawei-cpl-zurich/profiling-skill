# Gallery measurement ledger

Complete pairs: 28/28. Each complete arm passed one correctness call and ten profiled forwards with expected-kernel identity.

The original pair and full derived-evidence hashes in [gallery-evidence.json](gallery-evidence.json) bind these portable summaries to retained experiment artifacts. Machine routing/runtime configuration and full host metadata remain in those artifacts; these cases establish an A3-profile observation, not a result for every target.

Device span is last proven task end minus first task start, including inter-kernel gaps. Caller events are separately instrumented intervals. Busy time is a sum, active union is occupied time, and neither is elapsed span. No first sample is removed.

Gain/regression labels describe one capture per arm. They use a mean-difference interval with conservative t(9)=2.262 under mean/variance/independence assumptions; intervals crossing zero remain inconclusive. Repeated launches do not establish between-capture reproducibility. Hypothesis names identify the planned regimes, not their measured outcomes.

| Gallery case | Device span before / after (µs ± sample SD) | Span ratio and observation | Caller before / after (ms ± sample SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 01-triton-gallery-benefit | 26.396 ± 0.4543 / 11.274 ± 0.377 | 2.341×; gain | 0.224786 ± 0.07228 / 0.23505 ± 0.1124 | inconclusive |
| 01-triton-gallery-cost | 15.942 ± 0.6692 / 11.528 ± 0.4507 | 1.383×; gain | 0.320852 ± 0.3241 / 0.228294 ± 0.05785 | inconclusive |
| 02-triton-gallery-benefit | 103.472 ± 10.54 / 78.6245 ± 1.257 | 1.316×; gain | 0.316694 ± 0.08611 / 0.246434 ± 0.1198 | inconclusive |
| 02-catlass-gallery-benefit | 36.25 ± 6.975 / 31.242 ± 5.299 | 1.16×; inconclusive | 0.184558 ± 0.2222 / 0.137796 ± 0.07743 | inconclusive |
| 02-triton-gallery-cost | 87.1421 ± 11.85 / 57.5509 ± 1.084 | 1.514×; gain | 0.334488 ± 0.1524 / 0.208334 ± 0.09668 | inconclusive |
| 02-catlass-gallery-cost | 54.44 ± 4.259 / 52.336 ± 1.34 | 1.04×; inconclusive | 0.189488 ± 0.1125 / 0.161666 ± 0.1537 | inconclusive |
| 03-triton-gallery-benefit | 4537.93 ± 2.714 / 1455.64 ± 9.375 | 3.117×; gain | 4.57082 ± 0.1008 / 1.50654 ± 0.1576 | gain |
| 03-triton-gallery-cost | 20.184 ± 0.2372 / 10.9 ± 0.2451 | 1.852×; gain | 0.337766 ± 0.2815 / 0.224594 ± 0.08376 | inconclusive |
| 04-triton-gallery-benefit | 2429.02 ± 1.171 / 2027.27 ± 14.93 | 1.198×; gain | 2.47 ± 0.1283 / 2.10033 ± 0.2294 | gain |
| 04-triton-gallery-cost | 23.754 ± 1.27 / 186.132 ± 23.56 | 0.1276×; regression | 0.236746 ± 0.1305 / 0.430014 ± 0.1329 | regression |
| 05-triton-gallery-benefit | 152.131 ± 0.2516 / 122.294 ± 1.23 | 1.244×; gain | 0.327778 ± 0.1776 / 0.28848 ± 0.1108 | inconclusive |
| 05-catlass-gallery-benefit | 66.0186 ± 0.9286 / 35.9733 ± 1.551 | 1.835×; gain | 0.13734 ± 0.0787 / 0.127346 ± 0.05163 | inconclusive |
| 05-triton-gallery-cost | 274.912 ± 2.292 / 153.068 ± 1.746 | 1.796×; gain | 0.332506 ± 0.1254 / 0.282858 ± 0.07292 | inconclusive |
| 05-catlass-gallery-cost | 64.4647 ± 1.425 / 36.0293 ± 1.882 | 1.789×; gain | 0.199076 ± 0.1978 / 0.121868 ± 0.04899 | inconclusive |
| 06-catlass-gallery-benefit | 42.7411 ± 1.484 / 13.7797 ± 2.504 | 3.102×; gain | 0.14644 ± 0.1125 / 0.141998 ± 0.06016 | inconclusive |
| 06-catlass-gallery-cost | 15.196 ± 3.392 / 27.68 ± 3.898 | 0.549×; regression | 0.137164 ± 0.08454 / 0.20001 ± 0.1609 | inconclusive |
| 07-catlass-gallery-benefit | 315.436 ± 4.883 / 148.115 ± 1.624 | 2.13×; gain | 0.348434 ± 0.1011 / 0.221298 ± 0.0914 | gain |
| 07-catlass-gallery-cost | 5.54 ± 1.25 / 4.646 ± 0.8524 | 1.192×; inconclusive | 0.129332 ± 0.06487 / 0.13597 ± 0.08024 | inconclusive |
| 08-catlass-gallery-benefit | 24.468 ± 0.9446 / 22.436 ± 2.191 | 1.091×; gain | 0.132794 ± 0.08124 / 0.177662 ± 0.2231 | inconclusive |
| 08-catlass-gallery-cost | 6.54185 ± 2.791 / 6.01588 ± 0.7692 | 1.087×; inconclusive | 0.152598 ± 0.05258 / 0.177682 ± 0.2005 | inconclusive |
| 09-catlass-gallery-benefit | 15.7 ± 0.9159 / 15.322 ± 3.357 | 1.025×; inconclusive | 0.270156 ± 0.4029 / 0.172478 ± 0.1913 | inconclusive |
| 09-catlass-gallery-cost | 4.494 ± 0.9108 / 4.492 ± 0.8899 | 1×; inconclusive | 0.239518 ± 0.1523 / 0.21912 ± 0.2523 | inconclusive |
| 10-catlass-gallery-benefit | 40.284 ± 3.049 / 32.502 ± 1.95 | 1.239×; gain | 0.147674 ± 0.1247 / 0.20107 ± 0.2329 | inconclusive |
| 10-catlass-gallery-cost | 5.56 ± 2.385 / 12.484 ± 1.197 | 0.4454×; regression | 0.14381 ± 0.05379 / 0.140672 ± 0.1044 | inconclusive |
| 11-catlass-gallery-benefit | 145.947 ± 2.237 / 146.983 ± 4.339 | 0.993×; inconclusive | 0.283208 ± 0.2433 / 0.209842 ± 0.07063 | inconclusive |
| 11-catlass-gallery-cost | 6.62587 ± 3.523 / 5.77192 ± 1.537 | 1.148×; inconclusive | 0.230126 ± 0.1676 / 0.170388 ± 0.1953 | inconclusive |
| 12-triton-gallery-benefit | 1436.3 ± 9.048 / 1436.53 ± 4.622 | 0.9998×; inconclusive | 1.50274 ± 0.2151 / 1.4825 ± 0.1472 | inconclusive |
| 12-triton-gallery-cost | 3.85 ± 0.2598 / 4.222 ± 0.7394 | 0.9119×; inconclusive | 0.237184 ± 0.146 / 0.192532 ± 0.07155 | inconclusive |

All ten raw event/span samples, per-kernel raw PMU values, units, missing values, numerical diagnostics, workspace/resource settings and source/binary digests are in the JSON companion. Physical traffic and roofline distance are not inferred from activity percentages.

Resource increases remain costs even when timing improves. Several tested mechanisms have no demonstrated gain in these regimes; their recipes remain conditional hypotheses rather than universal recommendations.
