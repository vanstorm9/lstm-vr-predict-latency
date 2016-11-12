import numpy as np

def main(textFile="HandCoordinates.txt"):
    inputMatrix = []
    with open(textFile) as f:
        for line in f:
            inputMatrix.append(line.split())
    
    return np.array([[float(coord) for coord in frame] for frame in inputMatrix])   

if __name__ == "__main__":
    main()
