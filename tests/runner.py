import subprocess

def run():
    subprocess.run(
        ['poetry', 'run', 'python', '-u', '-m', 'unittest', 'discover']
    )