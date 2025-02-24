import numpy as np
import random

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)

    def append(self, samples):
        # appends at most `size` samples to the buffer, and returns how many samples were actually appended
        num_samples = len(samples)
        if num_samples >= self.size:
            self.buffer[:] = samples[:self.size] # copy the first `size` samples
            return self.size
        
        # shift the existing samples to the left to make space for all the new samples
        self.buffer[:-num_samples] = self.buffer[num_samples:]
        self.buffer[-num_samples:] = samples
        return num_samples
    

a = CircularBuffer(10)

sample_buffer = np.array([x for x in range(0, 100)], dtype=np.float32)
processed = 0
while processed < sample_buffer.size:
    num_samples = random.randint(1, 10)
    print(num_samples)
    processed += a.append(sample_buffer[processed:processed + num_samples])
    print(a.buffer)
    print()
