# Lilac_CCL 🚀

A hands-on journey through the most commonly used **Collective Communication** primitives — learn by doing!

Since all I have is a MacBook (no fancy GPU cluster here 😅), we simulate collective communication using the `gloo` backend. Don't let that stop you — the logic is exactly the same, and every concept translates directly to real multi-GPU environments.

Each primitive comes with a runnable test case. Just fire up the command below and watch the magic happen:
```bash
torchrun --nproc_per_node=4 xxx.py
```

## 🗺️ What's Coming

This repo isn't just about raw primitives. The goal is to build a **complete understanding** of how collective communication actually works in large-scale training frameworks.

Here's the roadmap:

- **Collective Primitives** — AllReduce, AllGather, ReduceScatter, AlltoAll, Broadcast, and more. Each with clean, runnable examples.
- **Megatron-LM Mappings** — We'll dive into `mappings.py` from Megatron-LM, where these primitives come to life in real `Tensor Parallel`、 `Pipeline Parallel`、 `Expert Parallel` training. friends will no longer be a mystery.
- **MoE Communication Patterns** — A closer look at how Mixture-of-Experts models dispatch and combine tokens across devices using AlltoAll and AllGather.

The idea is simple: **start from the basics, trace the code, connect the dots.**

## 💡 Why This Repo?

Most resources either stay too theoretical or dive straight into complex framework code. Here we try to meet in the middle — concrete examples first, then real-world mappings second.