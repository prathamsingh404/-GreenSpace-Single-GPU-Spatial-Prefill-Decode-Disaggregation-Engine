# Execution architecture

The engine creates a lattice of Green Context pairs, checks physical SM IDs for each pair, selects a pair, allocates shared model and KV tensors, orders initialization before worker access, and warms both execution paths. Each worker keeps its selected stream for the engine lifetime.

1. The scheduler admits a prompt to prefill. The block manager allocates physical pages, and the model writes prompt keys and values on the prefill stream.
2. Prefill samples the first token, records a CUDA event and publishes the request. A request that has reached its output limit releases its resources here.
3. Decode waits on the event, appends a token slot and runs a forward pass per active request. Decode gathers page contents using advanced indexing. Each greedy token is transferred to the host.
4. A completed or failed request releases its pages and event. Shutdown joins both workers before releasing the streams and marks unfinished requests as failed.

The model has RMSNorm, grouped-query attention and SwiGLU. Its current forward path uses initialized weights and lacks RoPE, a pretrained model loader and a reference-model equivalence test. The controller computes suggested workqueue limits and chunk sizes. The prefill worker processes a complete prompt per dispatch. Runtime temporal/spatial transitions, a calibrated roofline and continuous batched decode are design targets. [Checkpoint audit](AUDIT.md)
