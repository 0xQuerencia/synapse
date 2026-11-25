# Catalogue lock lifetime regression

Base revision: `e4aed57`. The final workspace test had reported `cache is busy`
immediately after `verify_model` returned. Inspection found that writer locks and
model pins relied on closing a `File` to release `flock` ownership. A duplicated
or inherited descriptor retains the same open file description, extending that
ownership beyond the intended Rust scope.

Two deterministic regressions retained duplicate descriptors after dropping the
writer guard or model pin. Before the fix both failed: the writer stayed busy,
and the released model remained protected from pruning. These reproduce a cause
consistent with the full-run failure; the exact child-spawn timing of that run
was not captured.

The fix explicitly unlocks each successful writer, shared pin and exclusive
prune guard when its owning process drops it. Acquisition remains nonblocking.
Live writers still refuse competing mutations, and independently acquired live
pins continue to protect models. A guard inherited by a forked child only closes
its copy, so dropping it cannot unlock the parent's live guard. An actual fork
regression checks this last property; the child performs only `getpid`, `close`
and `_exit` and does not run unrelated Rust destructors.

Executed on Linux with Rust 1.94.0:

```sh
cargo test --locked -p synapse-core --lib
```

Result: **234 passed, 0 failed, 0 ignored, 0 filtered out**. This includes the
previously failing catalogue test, both duplicate-descriptor regressions, the
forked-child regression, and the existing contention and live-pin tests. The
change does not retry cache operations, weaken lock acquisition, or remove lock
files to bypass contention. No application binary was rebuilt.
