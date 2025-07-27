import subprocess

def run_joern_query(query):
    joern_command = f'joern --script "query \"{query}\""'

    # Run Joern command
    process = subprocess.Popen(joern_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error running Joern command: {stderr.decode()}")
        return None

    # Process and return output
    result = stdout.decode()
    return result

# Example usage
if __name__ == "__main__":
    query = 'cpg.method.declarations.toList'
    result = run_joern_query(query)
    if result:
        print("Joern Query Result:")
        print(result)
