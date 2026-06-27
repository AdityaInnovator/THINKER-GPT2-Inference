// Global state for KV-cache attention (kernels/attention.cuh).
// Linked into all binaries that use the optimized attention path.

bool g_enable_kv_cache = false;
float* g_kv_cache = nullptr;
int g_layer_idx = 0;
bool g_is_prefill = true;
int g_current_pos = 0;
