#!/usr/bin/env python3
"""
Synthetic GPU Trace Generator for Trace-Driven GPU Memory Hierarchy Simulator
Produces deterministic synthetic traces for memory coalescing, cache hierarchy,
and warp scheduler experiments.
"""

import argparse
import random
import sys


def generate_trace(pattern: str,
                   num_warps: int,
                   num_instructions: int,
                   stride_words: int,
                   access_size: int,
                   alu_ratio: float,
                   seed: int,
                   output_path: str):
    random.seed(seed)
    lines = []
    lines.append(f"# Synthetic GPU Trace")
    lines.append(f"# Pattern: {pattern}")
    lines.append(f"# Warps: {num_warps}, Instructions/Warp: {num_instructions}")
    lines.append(f"# Stride: {stride_words} words, Access Size: {access_size} B, Seed: {seed}")
    lines.append(f"# Format: <OPCODE> <WARP_ID> <PC> [OPTIONS]")
    lines.append("")

    base_addresses = [0x10000000 + w * 0x100000 for w in range(num_warps)]
    current_offset = [0 for _ in range(num_warps)]

    # Matrix multiplication dimensions for matmul pattern (e.g., 64x64 float32 matrix)
    matrix_dim = 64
    element_size = 4

    for inst_idx in range(num_instructions):
        pc = 0x8000 + inst_idx * 4

        for warp_id in range(num_warps):
            # Decide whether this is an ALU or Memory instruction
            is_alu = (random.random() < alu_ratio)

            if is_alu:
                lines.append(f"ALU {warp_id} {hex(pc)}")
                continue

            # Memory instruction: alternate LOAD and STORE or pick LOAD
            op_type = "LOAD" if random.random() < 0.8 else "STORE"
            mask_hex = "FFFFFFFF"  # All 32 lanes active

            if pattern == "sequential":
                # Contiguous 32 words (128 bytes total per warp)
                base = base_addresses[warp_id] + current_offset[warp_id]
                current_offset[warp_id] += 32 * access_size
                lines.append(f"{op_type}_STRIDED {warp_id} {hex(pc)} {mask_hex} {access_size} {hex(base)} {access_size}")

            elif pattern == "strided":
                # Stride in bytes
                stride_bytes = stride_words * access_size
                base = base_addresses[warp_id] + current_offset[warp_id]
                current_offset[warp_id] += 32 * stride_bytes
                lines.append(f"{op_type}_STRIDED {warp_id} {hex(pc)} {mask_hex} {access_size} {hex(base)} {stride_bytes}")

            elif pattern == "random":
                # 32 random addresses within a 256 KB memory range
                range_size = 256 * 1024
                addrs = [base_addresses[warp_id] + (random.randint(0, range_size - 1) & ~0x3) for _ in range(32)]
                addr_str = " ".join(hex(a) for a in addrs)
                lines.append(f"{op_type} {warp_id} {hex(pc)} {mask_hex} {access_size} {addr_str}")

            elif pattern == "repeated":
                # Repeatedly accesses a small working set of 8 cache lines (1024 bytes)
                working_set_lines = 8
                line_idx = (inst_idx % working_set_lines)
                base = base_addresses[warp_id] + line_idx * 128
                lines.append(f"{op_type}_STRIDED {warp_id} {hex(pc)} {mask_hex} {access_size} {hex(base)} {access_size}")

            elif pattern == "streaming":
                # Large sweep with zero temporal reuse, strictly contiguous forward strides
                base = base_addresses[warp_id] + current_offset[warp_id]
                current_offset[warp_id] += 32 * access_size
                lines.append(f"{op_type}_STRIDED {warp_id} {hex(pc)} {mask_hex} {access_size} {hex(base)} {access_size}")

            elif pattern == "matmul":
                # Simulates tile read: Lane i reads B[k, i] (strided by matrix_dim * element_size)
                # while reading A[row, k] (uniform across all lanes)
                k = inst_idx % matrix_dim
                row = (warp_id * 16 + (inst_idx // matrix_dim)) % matrix_dim
                
                # Step 1: Read row element of A (uniform load broadcast to all lanes)
                addr_a = 0x20000000 + (row * matrix_dim + k) * element_size
                lines.append(f"LOAD_UNIFORM {warp_id} {hex(pc)} {mask_hex} {element_size} {hex(addr_a)}")

                # Step 2: Read column elements of B: Lane i reads B[k, lane_i] (unit stride along row k)
                addr_b_base = 0x30000000 + (k * matrix_dim) * element_size
                lines.append(f"LOAD_STRIDED {warp_id} {hex(pc + 4)} {mask_hex} {element_size} {hex(addr_b_base)} {element_size}")

                # Step 3: Compute FMA
                lines.append(f"ALU {warp_id} {hex(pc + 8)}")
            else:
                raise ValueError(f"Unknown workload pattern: {pattern}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[SUCCESS] Generated {len(lines)} lines to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic GPU Trace Generator")
    parser.add_argument("--pattern", type=str, default="sequential",
                        choices=["sequential", "strided", "random", "repeated", "streaming", "matmul"],
                        help="Memory access pattern")
    parser.add_argument("--warps", type=int, default=4, help="Number of active warps")
    parser.add_argument("--instructions", type=int, default=50, help="Number of instructions per warp")
    parser.add_argument("--stride", type=int, default=1, help="Stride in words (for strided pattern)")
    parser.add_argument("--access-size", type=int, default=4, help="Access size in bytes per thread")
    parser.add_argument("--alu-ratio", type=float, default=0.2, help="Ratio of ALU instructions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--output", type=str, default="traces/sample.trace", help="Output file path")

    args = parser.parse_args()
    generate_trace(
        pattern=args.pattern,
        num_warps=args.warps,
        num_instructions=args.instructions,
        stride_words=args.stride,
        access_size=args.access_size,
        alu_ratio=args.alu_ratio,
        seed=args.seed,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
