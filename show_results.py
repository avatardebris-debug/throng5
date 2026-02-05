with open('test_results_summary.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    
# Print each line
for line in content.split('\n'):
    print(line)
