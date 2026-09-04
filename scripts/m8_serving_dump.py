# NEXT-SESSION HARNESS: dump the fused kernel's in-graph x/y at verify time.
# In ggml_dense._kq_gemv's fused branch (FREETOKEN_SPEC_DBG=1), stash
# (w, x, y) into module globals; at the scheduler's first replay (after
# stream sync) re-run kq_gemm_q4k_m8(w, x) EAGERLY and diff against the
# stashed y. A diff => the captured launch is unfaithful (stream/pool);
# no diff => the corruption is downstream of this kernel.
# Kernel status: BIT-EXACT offline vs the T=1 GEMV oracle on real weights
# (scripts/m8_parity.py) AND under isolated graph capture+replay
# (scripts/m8_graph_repro.py). Only the serving-graph path diverges.
