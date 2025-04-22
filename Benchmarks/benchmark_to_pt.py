import os
import argparse
import numpy as np
import torch
import gzip
import glob
from tqdm import tqdm


def convert_gtsp_to_pt(input_file, output_file):
    """
    Convert a GTSP instance (possibly gzipped) to PyTorch tensor format

    Args:
        input_file (str): Path to the input GTSP file (gzipped or not)
        output_file (str): Path to the output PT file
    """
    print(f"Converting {input_file} to {output_file}...")

    # Determine if the file is gzipped
    is_gzipped = input_file.endswith('.gz')

    # Read the input file
    if is_gzipped:
        with gzip.open(input_file, 'rt') as f:
            lines = f.readlines()
    else:
        with open(input_file, 'r') as f:
            lines = f.readlines()

    # Parse the header information
    dimension = None
    num_sets = None

    for line in lines:
        if 'DIMENSION' in line:
            dimension = int(line.split(':')[1].strip())
        elif 'GTSP_SETS' in line:
            num_sets = int(line.split(':')[1].strip())
        elif 'NODE_COORD_SECTION' in line:
            break

    if dimension is None or num_sets is None:
        raise ValueError(f"Could not find DIMENSION or GTSP_SETS in the input file: {input_file}")

    print(f"Dimension: {dimension}, Number of Sets: {num_sets}")

    # Parse the node coordinates
    node_coords = []
    node_section_started = False
    sets_section_started = False

    # Initialize cluster assignments to -1 (unassigned)
    node_clusters = [-1] * (dimension + 1)  # +1 because nodes are 1-indexed

    for line in lines:
        line = line.strip()

        if line == 'NODE_COORD_SECTION':
            node_section_started = True
            continue
        elif line == 'GTSP_SET_SECTION:':
            node_section_started = False
            sets_section_started = True
            continue
        elif line == 'EOF':
            break

        if node_section_started and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append((node_id, x, y))

        elif sets_section_started and line:
            parts = line.split()
            if len(parts) >= 2:
                cluster_id = int(parts[0]) - 1  # Convert to 0-indexed for lkh.py

                # Assign all nodes in this set to the cluster
                for node_id_str in parts[1:]:
                    if node_id_str == '-1':
                        break
                    node_id = int(node_id_str)
                    node_clusters[node_id] = cluster_id

    # Make sure all nodes have been assigned to a cluster
    if -1 in node_clusters[1:]:
        unassigned = [i for i, c in enumerate(node_clusters) if c == -1 and i > 0]
        raise ValueError(f"Some nodes were not assigned to a cluster: {unassigned} in file {input_file}")

    # Sort nodes by ID to ensure correct order
    node_coords.sort(key=lambda x: x[0])

    # Extract coordinates in correct order
    coords = [(x, y) for _, x, y in node_coords]

    # Ensure we have the right number of nodes
    if len(coords) != dimension:
        raise ValueError(f"Expected {dimension} nodes, but found {len(coords)} in file {input_file}")

    # Create tensors
    node_xy = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # [1, problem_size, 2]
    cluster_idx = torch.tensor(node_clusters[1:], dtype=torch.long).unsqueeze(0)  # [1, problem_size]

    # Save tensors
    torch.save({
        'node_xy': node_xy,
        'cluster_idx': cluster_idx
    }, output_file)

    print(f"Converted file saved to {output_file}")
    return True


def batch_convert(input_pattern, output_dir):
    """
    Convert multiple GTSP files matching a pattern to PT format

    Args:
        input_pattern (str): Glob pattern for input files
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all files matching the pattern
    input_files = glob.glob(input_pattern)

    if not input_files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    print(f"Found {len(input_files)} files to convert")

    # Convert each file
    success_count = 0
    for input_file in tqdm(input_files):
        try:
            # Generate output filename
            basename = os.path.basename(input_file)
            basename_noext = os.path.splitext(basename)[0]

            # If the file is gzipped, we need to remove two extensions
            if basename_noext.endswith('.gtsp'):
                basename_noext = os.path.splitext(basename_noext)[0]

            output_file = os.path.join(output_dir, f"{basename_noext}.pt")

            # Convert the file
            if convert_gtsp_to_pt(input_file, output_file):
                success_count += 1

        except Exception as e:
            print(f"Error converting {input_file}: {e}")

    print(f"Successfully converted {success_count} out of {len(input_files)} files")


def main():
    parser = argparse.ArgumentParser(description='Convert GTSP instances to PyTorch tensor format')
    parser.add_argument('--input', '-i', type=str, help='Input GTSP file or glob pattern')
    parser.add_argument('--output_dir', '-o', type=str, default='converted',
                        help='Output directory for converted files')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch process multiple files matching a glob pattern')
    parser.add_argument('--single', '-s', action='store_true',
                        help='Force single file conversion mode')

    args = parser.parse_args()

    # 如果没有指定输入文件，或者明确指定了批处理模式
    if (not args.input and not args.single) or args.batch:
        # 默认使用批处理模式
        input_pattern = args.input if args.input else "*.gtsp.gz"
        print(f"Using batch mode with pattern: {input_pattern}")
        batch_convert(input_pattern, args.output_dir)
    else:
        # 单文件模式
        if not args.input:
            print("Error: Input file must be specified for single file conversion")
            print("Use --batch to process all *.gtsp.gz files or specify an input file with --input")
            return

        # Generate output filename
        basename = os.path.basename(args.input)
        basename_noext = os.path.splitext(basename)[0]

        # If the file is gzipped, we need to remove two extensions
        if basename_noext.endswith('.gtsp'):
            basename_noext = os.path.splitext(basename_noext)[0]

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{basename_noext}.pt")

        # Convert the file
        convert_gtsp_to_pt(args.input, output_file)


if __name__ == "__main__":
    main()