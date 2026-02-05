import re
import sys

# Redirect to file
output_lines = []

# Read transfer test output
try:
    with open('transfer_output.log', 'rb') as f:
        transfer_text = f.read().decode('utf-16-le', errors='ignore')
    
    output_lines.append("=" * 60)
    output_lines.append("TRANSFER TEST RESULTS")
    output_lines.append("=" * 60)
    
    # Extract key metrics
    task_a_match = re.search(r'Task A final loss:\s+([\d.]+)', transfer_text)
    task_b_transfer_match = re.search(r'Task B \(transfer\):\s+([\d.]+)', transfer_text)
    task_b_fresh_match = re.search(r'Task B \(fresh\):\s+([\d.]+)', transfer_text)
    improvement_match = re.search(r'Transfer improvement:\s+([-\d.]+)%', transfer_text)
    
    if task_a_match:
        output_lines.append(f"Task A final loss: {task_a_match.group(1)}")
    if task_b_transfer_match:
        output_lines.append(f"Task B (transfer): {task_b_transfer_match.group(1)}")
    if task_b_fresh_match:
        output_lines.append(f"Task B (fresh):    {task_b_fresh_match.group(1)}")
    if improvement_match:
        output_lines.append(f"Transfer improvement: {improvement_match.group(1)}%")
    
    # Extract meta-level comparison
    output_lines.append("")
    output_lines.append("Meta-level comparison:")
    for match in re.finditer(r'Meta\^0-(\d+) \((\d+) layers\): transfer loss=([\d.]+)', transfer_text):
        output_lines.append(f"  {match.group(2)} layers: {match.group(3)}")
    
except Exception as e:
    output_lines.append(f"Error reading transfer test: {e}")

output_lines.append("")

# Read superexponential test output
try:
    with open('super_output.log', 'rb') as f:
        super_text = f.read().decode('utf-16-le', errors='ignore')
    
    output_lines.append("=" * 60)
    output_lines.append("SUPEREXPONENTIAL TEST RESULTS")
    output_lines.append("=" * 60)
    
    # Extract layer performance
    for match in re.finditer(r'Meta\^0-(\d+) \((\d+) layers\): final_loss=([\d.]+), signals=(\d+)', super_text):
        output_lines.append(f"{match.group(2)} layers: loss={match.group(3)}, signals={match.group(4)}")
    
    # Extract improvements
    improvements_match = re.search(r'Improvements by adding each layer: \[(.*?)\]', super_text)
    if improvements_match:
        output_lines.append(f"\nImprovements: [{improvements_match.group(1)}]")
    
    accel_match = re.search(r'Acceleration: \[(.*?)\]', super_text)
    if accel_match:
        output_lines.append(f"Acceleration: [{accel_match.group(1)}]")
    
    superexp_match = re.search(r'Superexponential: (.+)', super_text)
    if superexp_match:
        output_lines.append(f"Superexponential: {superexp_match.group(1)}")
    
except Exception as e:
    output_lines.append(f"Error reading superexponential test: {e}")

# Write to file
with open('test_results_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("Results written to test_results_summary.txt")
