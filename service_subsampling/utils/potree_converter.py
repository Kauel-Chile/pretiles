import subprocess

def run_potree_converter(input_file: str, output_file: str):
    command = f"/home/opengeo/PotreeConverter/build/PotreeConverter {input_file} -o {output_file}"
    print(command)
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command output:", result.stdout.decode())
        print("Command error (if any):", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the command:", e)