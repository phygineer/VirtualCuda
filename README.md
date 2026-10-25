# VirtualCuda

> Status: idea stage

## The Idea

> "We do not need big hardware for LLM if we can just generate layer outputs by predicting it."

> "What CUDA does can be just a higher dimensional math function, and that's why VirtualCuda."

Each layer of an LLM is a math function: a high-dimensional vector goes in, a vector comes out. CUDA and big GPUs are just a way to compute those functions exactly, fast.

VirtualCuda explores whether those layer outputs can be **predicted** instead of computed exactly, so LLMs can run without big hardware.
