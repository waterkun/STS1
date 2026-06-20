from pathlib import Path
import os



RUNS_DIR = Path(__file__).parent.parent/'runs'



def load_ironclad_runs()->list[dict]:
    print("start to load runs")
    runs = []
    os.walk(RUNS_DIR)
    return runs



def main():
    runs = []
    runs = load_ironclad_runs()

if __name__=='__main__':
    main()

